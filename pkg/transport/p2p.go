// Package transport provides P2P communication using libp2p.
package transport

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	libp2pprotocol "github.com/libp2p/go-libp2p/core/protocol"
	"github.com/libp2p/go-libp2p/p2p/discovery/mdns"
	"github.com/multiformats/go-multiaddr"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/neurogrid/kv-cache-p2p/pkg/cache"
	"github.com/neurogrid/kv-cache-p2p/pkg/protocol"
)

const (
	// ServiceTag for mDNS discovery
	ServiceTag = "neurogrid-kvcache"

	// Default timeouts
	DefaultLookupTimeout  = 5 * time.Second
	DefaultTransferTimeout = 30 * time.Second
)

// P2PNode manages libp2p communication for KV cache sharing.
type P2PNode struct {
	host     host.Host
	storage  cache.Storage
	peers    map[peer.ID]*PeerState
	peersMu  sync.RWMutex
	ctx      context.Context
	cancel   context.CancelFunc
	reqIDGen atomic.Uint64
}

// PeerState tracks state of a connected peer.
type PeerState struct {
	ID         peer.ID
	Addrs      []multiaddr.Multiaddr
	LastSeen   time.Time
	CacheStats protocol.Stats
	Connected  bool
}

// response wraps response data.
type response struct {
	header  protocol.Header
	payload []byte
	err     error
}

// Config holds P2P node configuration.
type Config struct {
	ListenPort     int
	EnableMDNS     bool
	BootstrapPeers []string // Multiaddrs of bootstrap peers
	ExternalIP     string   // External/public IP to announce (optional)
}

// NewP2PNode creates a new P2P node.
func NewP2PNode(ctx context.Context, cfg Config, storage cache.Storage) (*P2PNode, error) {
	nodeCtx, cancel := context.WithCancel(ctx)

	// Create listen address
	listenAddr, err := multiaddr.NewMultiaddr(fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", cfg.ListenPort))
	if err != nil {
		cancel()
		return nil, fmt.Errorf("invalid listen address: %w", err)
	}

	// Build libp2p options
	opts := []libp2p.Option{
		libp2p.ListenAddrs(listenAddr),
	}

	// If external IP is specified, announce it
	if cfg.ExternalIP != "" {
		externalAddr, err := multiaddr.NewMultiaddr(fmt.Sprintf("/ip4/%s/tcp/%d", cfg.ExternalIP, cfg.ListenPort))
		if err != nil {
			cancel()
			return nil, fmt.Errorf("invalid external address: %w", err)
		}
		opts = append(opts, libp2p.AddrsFactory(func(addrs []multiaddr.Multiaddr) []multiaddr.Multiaddr {
			// Add external address to the list
			return append(addrs, externalAddr)
		}))
		log.Printf("Announcing external address: %s", externalAddr)
	}

	// Create libp2p host
	h, err := libp2p.New(opts...)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create host: %w", err)
	}

	node := &P2PNode{
		host:    h,
		storage: storage,
		peers:   make(map[peer.ID]*PeerState),
		ctx:     nodeCtx,
		cancel:  cancel,
	}

	// Register protocol handler
	h.SetStreamHandler(libp2pprotocol.ID(protocol.ProtocolID), node.handleStream)

	// Setup mDNS if enabled
	if cfg.EnableMDNS {
		notifee := &discoveryNotifee{node: node}
		mdnsService := mdns.NewMdnsService(h, ServiceTag, notifee)
		if err := mdnsService.Start(); err != nil {
			log.Printf("Warning: mDNS start failed: %v", err)
		}
	}

	// Connect to bootstrap peers
	for _, addr := range cfg.BootstrapPeers {
		if err := node.ConnectPeer(ctx, addr); err != nil {
			log.Printf("Warning: failed to connect to bootstrap peer %s: %v", addr, err)
		}
	}

	log.Printf("P2P node started: %s", h.ID())
	for _, addr := range h.Addrs() {
		log.Printf("  Listening: %s/p2p/%s", addr, h.ID())
	}

	return node, nil
}

// ConnectPeer connects to a peer by multiaddr and adds it to the peer list.
func (n *P2PNode) ConnectPeer(ctx context.Context, addr string) error {
	ma, err := multiaddr.NewMultiaddr(addr)
	if err != nil {
		return err
	}

	pi, err := peer.AddrInfoFromP2pAddr(ma)
	if err != nil {
		return err
	}

	if err := n.host.Connect(ctx, *pi); err != nil {
		return err
	}

	// Add to peer list after successful connection
	n.peersMu.Lock()
	n.peers[pi.ID] = &PeerState{
		ID:        pi.ID,
		Addrs:     pi.Addrs,
		LastSeen:  time.Now(),
		Connected: true,
	}
	n.peersMu.Unlock()

	log.Printf("Connected to peer: %s", pi.ID)
	return nil
}

// handleStream processes incoming streams (requests from other nodes).
func (n *P2PNode) handleStream(s network.Stream) {
	defer s.Close()

	// Read message
	header, payload, err := protocol.ReadMessage(s)
	if err != nil {
		log.Printf("Error reading message from %s: %v", s.Conn().RemotePeer(), err)
		return
	}

	// Handle incoming request and write response on same stream
	switch header.Type {
	case protocol.MsgLookup:
		n.handleLookup(s, header, payload)
	case protocol.MsgGet:
		n.handleGet(s, header, payload)
	case protocol.MsgPut:
		n.handlePut(s, header, payload)
	case protocol.MsgPing:
		n.handlePing(s, header, payload)
	default:
		log.Printf("Unknown message type: %d", header.Type)
	}
}

// handleLookup processes lookup requests.
func (n *P2PNode) handleLookup(s network.Stream, header protocol.Header, payload []byte) {
	var req protocol.LookupRequest
	if err := msgpack.Unmarshal(payload, &req); err != nil {
		return
	}

	resp := protocol.LookupResponse{
		Found: make([]bool, len(req.Keys)),
		Sizes: make([]int64, len(req.Keys)),
	}

	for i, keyWire := range req.Keys {
		key := wireToKey(keyWire)
		if n.storage.Contains(n.ctx, key) {
			resp.Found[i] = true
			if entry, err := n.storage.Get(n.ctx, key); err == nil {
				resp.Sizes[i] = int64(entry.Size())
				entry.Unref()
			}
		}
	}

	protocol.WriteMessage(s, protocol.MsgLookupAck, header.RequestID, resp)
}

// handleGet processes get requests.
func (n *P2PNode) handleGet(s network.Stream, header protocol.Header, payload []byte) {
	var req protocol.GetRequest
	if err := msgpack.Unmarshal(payload, &req); err != nil {
		return
	}

	resp := protocol.GetResponse{
		Entries: make([]protocol.CacheEntryWire, 0, len(req.Keys)),
		Errors:  make([]string, len(req.Keys)),
	}

	for i, keyWire := range req.Keys {
		key := wireToKey(keyWire)
		entry, err := n.storage.Get(n.ctx, key)
		if err != nil {
			resp.Errors[i] = err.Error()
			continue
		}

		resp.Entries = append(resp.Entries, protocol.CacheEntryWire{
			Key:       keyWire,
			K:         entry.K,
			V:         entry.V,
			CreatedAt: entry.CreatedAt.UnixNano(),
		})
		entry.Unref()
	}

	protocol.WriteMessage(s, protocol.MsgGetAck, header.RequestID, resp)
}

// handlePut processes put requests.
func (n *P2PNode) handlePut(s network.Stream, header protocol.Header, payload []byte) {
	var req protocol.PutRequest
	if err := msgpack.Unmarshal(payload, &req); err != nil {
		return
	}

	resp := protocol.PutResponse{
		Accepted: make([]bool, len(req.Entries)),
		Errors:   make([]string, len(req.Entries)),
	}

	for i, entryWire := range req.Entries {
		key := wireToKey(entryWire.Key)
		entry := cache.NewCacheEntry(key, entryWire.K, entryWire.V)

		if err := n.storage.Put(n.ctx, entry); err != nil {
			resp.Errors[i] = err.Error()
		} else {
			resp.Accepted[i] = true
		}
	}

	protocol.WriteMessage(s, protocol.MsgPutAck, header.RequestID, resp)
}

// handlePing processes ping requests.
func (n *P2PNode) handlePing(s network.Stream, header protocol.Header, payload []byte) {
	var req protocol.PingRequest
	if err := msgpack.Unmarshal(payload, &req); err != nil {
		return
	}

	stats := n.storage.Stats()
	hitRate := float64(0)
	if total := stats.HitCount + stats.MissCount; total > 0 {
		hitRate = float64(stats.HitCount) / float64(total)
	}

	resp := protocol.PongResponse{
		SentAt:     req.SentAt,
		ReceivedAt: time.Now().UnixNano(),
		CacheStats: protocol.Stats{
			Entries:   stats.Entries,
			SizeBytes: stats.SizeBytes,
			HitRate:   hitRate,
		},
	}

	protocol.WriteMessage(s, protocol.MsgPong, header.RequestID, resp)
}

// Lookup queries peers for cache keys.
func (n *P2PNode) Lookup(ctx context.Context, keys []cache.CacheKey) (map[peer.ID][]int, error) {
	wireKeys := make([]protocol.CacheKeyWire, len(keys))
	for i, k := range keys {
		wireKeys[i] = keyToWire(k)
	}

	req := protocol.LookupRequest{Keys: wireKeys}
	reqID := n.reqIDGen.Add(1)

	results := make(map[peer.ID][]int)
	var wg sync.WaitGroup

	n.peersMu.RLock()
	peers := make([]peer.ID, 0, len(n.peers))
	for pid := range n.peers {
		peers = append(peers, pid)
	}
	n.peersMu.RUnlock()

	var resultsMu sync.Mutex

	for _, pid := range peers {
		wg.Add(1)
		go func(pid peer.ID) {
			defer wg.Done()

			resp, err := n.sendRequest(ctx, pid, protocol.MsgLookup, reqID, req)
			if err != nil {
				return
			}

			var lookupResp protocol.LookupResponse
			if err := msgpack.Unmarshal(resp.payload, &lookupResp); err != nil {
				return
			}

			var foundIdxs []int
			for i, found := range lookupResp.Found {
				if found {
					foundIdxs = append(foundIdxs, i)
				}
			}

			if len(foundIdxs) > 0 {
				resultsMu.Lock()
				results[pid] = foundIdxs
				resultsMu.Unlock()
			}
		}(pid)
	}

	wg.Wait()
	return results, nil
}

// FetchFromPeer retrieves cache entries from a specific peer.
func (n *P2PNode) FetchFromPeer(ctx context.Context, pid peer.ID, keys []cache.CacheKey) ([]*cache.CacheEntry, error) {
	wireKeys := make([]protocol.CacheKeyWire, len(keys))
	for i, k := range keys {
		wireKeys[i] = keyToWire(k)
	}

	req := protocol.GetRequest{Keys: wireKeys}
	reqID := n.reqIDGen.Add(1)

	resp, err := n.sendRequest(ctx, pid, protocol.MsgGet, reqID, req)
	if err != nil {
		return nil, err
	}

	var getResp protocol.GetResponse
	if err := msgpack.Unmarshal(resp.payload, &getResp); err != nil {
		return nil, err
	}

	entries := make([]*cache.CacheEntry, len(getResp.Entries))
	for i, ew := range getResp.Entries {
		key := wireToKey(ew.Key)
		entries[i] = cache.NewCacheEntry(key, ew.K, ew.V)
	}

	return entries, nil
}

// sendRequest sends a request and waits for response on the same stream.
func (n *P2PNode) sendRequest(ctx context.Context, pid peer.ID, msgType protocol.MessageType, reqID uint64, payload interface{}) (*response, error) {
	// Open stream
	s, err := n.host.NewStream(ctx, pid, libp2pprotocol.ID(protocol.ProtocolID))
	if err != nil {
		return nil, fmt.Errorf("failed to open stream: %w", err)
	}
	defer s.Close()

	// Send request
	if err := protocol.WriteMessage(s, msgType, reqID, payload); err != nil {
		return nil, fmt.Errorf("failed to write message: %w", err)
	}

	// Read response from the same stream (handlers write back on same stream)
	respChan := make(chan *response, 1)
	errChan := make(chan error, 1)

	go func() {
		header, respPayload, err := protocol.ReadMessage(s)
		if err != nil {
			errChan <- fmt.Errorf("failed to read response: %w", err)
			return
		}
		respChan <- &response{header: header, payload: respPayload}
	}()

	// Wait for response or timeout
	select {
	case resp := <-respChan:
		return resp, nil
	case err := <-errChan:
		return nil, err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Peers returns list of connected peers.
func (n *P2PNode) Peers() []PeerState {
	n.peersMu.RLock()
	defer n.peersMu.RUnlock()

	peers := make([]PeerState, 0, len(n.peers))
	for _, p := range n.peers {
		peers = append(peers, *p)
	}
	return peers
}

// Close shuts down the node.
func (n *P2PNode) Close() error {
	n.cancel()
	return n.host.Close()
}

// Host returns the underlying libp2p host.
func (n *P2PNode) Host() host.Host {
	return n.host
}

// discoveryNotifee handles mDNS discovery.
type discoveryNotifee struct {
	node *P2PNode
}

func (d *discoveryNotifee) HandlePeerFound(pi peer.AddrInfo) {
	log.Printf("Discovered peer: %s", pi.ID)

	if err := d.node.host.Connect(d.node.ctx, pi); err != nil {
		log.Printf("Failed to connect to discovered peer: %v", err)
		return
	}

	d.node.peersMu.Lock()
	d.node.peers[pi.ID] = &PeerState{
		ID:        pi.ID,
		Addrs:     pi.Addrs,
		LastSeen:  time.Now(),
		Connected: true,
	}
	d.node.peersMu.Unlock()

	log.Printf("Connected to peer: %s", pi.ID)
}

// Helper functions for wire conversion.
func keyToWire(k cache.CacheKey) protocol.CacheKeyWire {
	return protocol.CacheKeyWire{
		ModelID:   k.ModelID,
		LayerID:   k.LayerID,
		TokenHash: k.TokenHash[:],
		SeqStart:  k.SeqStart,
		SeqEnd:    k.SeqEnd,
	}
}

func wireToKey(w protocol.CacheKeyWire) cache.CacheKey {
	var hash [32]byte
	copy(hash[:], w.TokenHash)
	return cache.CacheKey{
		ModelID:   w.ModelID,
		LayerID:   w.LayerID,
		TokenHash: hash,
		SeqStart:  w.SeqStart,
		SeqEnd:    w.SeqEnd,
	}
}
