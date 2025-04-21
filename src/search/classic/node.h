/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors
  ... (License header) ...
*/

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

// Corrected Includes (Using chess/ prefix)
#include "chess/board.h"     // For MoveList definition
#include "chess/callbacks.h" // Needed indirectly? Keep for safety.
#include "chess/types.h"     // <<< Defines lczero::Value, lczero::GameResult, lczero::kValueMate, lczero::Move etc.
#include "chess/gamestate.h" // For PositionHistory
#include "chess/position.h" // For PositionHash and GameResult? (Confirm GameResult source)
#include "neural/encoder.h"
#include "proto/net.pb.h"   // For EvalResult
#include "utils/mutex.h"

namespace lczero {
namespace classic {

// Forward declarations
class SearchParams;
class Node;
class EdgeAndNode;
template <bool is_const> class Edge_Iterator;
template <bool is_const> class VisitedNode_Iterator;


class Edge {
 public:
  static std::unique_ptr<Edge[]> FromMovelist(const lczero::MoveList& moves); // Use lczero::
  lczero::Move GetMove(bool as_opponent = false) const; // Use lczero::
  float GetP() const;
  void SetP(float val);
  std::string DebugString() const;

 private:
  lczero::Move move_;
  uint16_t p_ = 0;
  friend class Node;
  friend class Edge_Iterator<true>;
  friend class Edge_Iterator<false>;
};

class Node {
 public:
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;

  enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase, TwoFold };

  Node(Node* parent, uint16_t index);

  Node(Node&& move_from) = default;
  Node& operator=(Node&& move_from) = default;
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  Node* CreateSingleChildNode(lczero::Move m); // Use lczero::
  void CreateEdges(const lczero::MoveList& moves); // Use lczero::
  Node* GetParent() const { return parent_; }
  bool HasChildren() const { return static_cast<bool>(edges_); }
  float GetVisitedPolicy() const;

  uint32_t GetN() const { return n_.load(std::memory_order_relaxed); }
  uint32_t GetNInFlight() const { return n_in_flight_.load(std::memory_order_relaxed); }
  uint32_t GetVisitCount() const { return GetN(); }
  int GetNStarted() const { return n_.load(std::memory_order_relaxed) + n_in_flight_.load(std::memory_order_relaxed); }
  uint32_t GetChildrenVisits() const {
      uint32_t n = GetN();
      return n > 0 ? n - 1 : 0;
  }
  int GetEffectiveVisits() const { return GetNStarted(); }
  int GetEffectiveParentVisits() const;

  float GetQ(float draw_score) const;
  lczero::Value GetValue() const; // <<< Use lczero::Value
  float GetWL() const { return static_cast<float>(wl_.load(std::memory_order_relaxed)); }
  float GetD() const { return d_.load(std::memory_order_relaxed); }
  float GetM() const { return m_.load(std::memory_order_relaxed); }
  float GetPolicyPrior() const;

  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  bool IsTwoFoldTerminal() const { return terminal_type_ == Terminal::TwoFold; }
  typedef std::pair<lczero::GameResult, lczero::GameResult> Bounds; // <<< Use lczero::GameResult
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  void SetBounds(lczero::GameResult lower, lczero::GameResult upper); // <<< Use lczero::GameResult
  void MakeTerminal(lczero::GameResult result, float plies_left = 0.0f, // <<< Use lczero::GameResult
                    Terminal type = Terminal::EndOfGame);
  void MakeNotTerminal();

  uint8_t GetNumEdges() const { return num_edges_; }
  void CopyPolicy(int max_needed, float* output) const;

  bool TryStartScoreUpdate();
  void CancelScoreUpdate(int multivisit);
  void FinalizeScoreUpdate(lczero::Value v, float d, float m, int multivisit); // <<< Use lczero::Value
  void AdjustForTerminal(lczero::Value v_delta, float d_delta, float m_delta, int multivisit); // <<< Use lczero::Value
  void RevertTerminalVisits(lczero::Value v, float d, float m, int multivisit); // <<< Use lczero::Value
  void IncrementNInFlight(int multivisit) { n_in_flight_.fetch_add(multivisit, std::memory_order_relaxed); }

  ConstIterator Edges() const;
  Iterator Edges();
  VisitedNode_Iterator<true> VisitedNodes() const;
  VisitedNode_Iterator<false> VisitedNodes();

  void ReleaseChildren();
  void ReleaseChildrenExceptOne(Node* node_to_save);
  Edge* GetEdgeToNode(const Node* node) const;
  Edge* GetOwnEdge() const;
  std::string DebugString() const;
  bool MakeSolid();
  void SortEdges();

  uint16_t Index() const { return index_; }

   std::atomic<bool> is_known_win{false};
   std::atomic<bool> is_known_loss{false};

   lczero::Value GetMinValue() const; // <<< Use lczero::Value
   lczero::Value GetMaxValue() const; // <<< Use lczero::Value

   const std::unique_ptr<Node>* GetChildrenPtr() const { return &child_; }
   Node* GetChild(int index) const;
   int GetNumChildren() const { return num_edges_; }

  ~Node();

 private:
  void UpdateChildrenParents();

  std::unique_ptr<Edge[]> edges_;
  Node* parent_ = nullptr;
  std::unique_ptr<Node> child_;
  std::unique_ptr<Node> sibling_;
  std::atomic<double> wl_{0.0};

  std::atomic<float> d_{0.0};
  std::atomic<float> m_{0.0};
  std::atomic<uint32_t> n_{0};
  std::atomic<uint32_t> n_in_flight_{0};

  uint16_t index_;

  uint8_t num_edges_ = 0;
  Terminal terminal_type_ : 2;
  lczero::GameResult lower_bound_ : 2; // <<< Use lczero::GameResult
  lczero::GameResult upper_bound_ : 2; // <<< Use lczero::GameResult
  bool solid_children_ : 1;

  friend class NodeTree;
  friend class Edge_Iterator<true>;
  friend class Edge_Iterator<false>;
  friend class VisitedNode_Iterator<true>;
  friend class VisitedNode_Iterator<false>;
  friend class SearchWorker;
};

// --- EdgeAndNode ---
class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : edge_(edge), node_(node) {}
  void Reset() { edge_ = nullptr; node_ = nullptr; }
  explicit operator bool() const { return edge_ != nullptr; }
   bool operator==(const EdgeAndNode& other) const {
       return edge_ == other.edge_ && node_ == other.node_;
   }
   bool operator!=(const EdgeAndNode& other) const { return !(*this == other); }
   bool HasNode() const { return node_ != nullptr; }
   Edge* edge() const { return edge_; }
   Node* node() const { return node_; }

   float GetQ(float default_q, float draw_score) const;
   float GetWL(float default_wl) const;
   float GetD(float default_d) const;
   float GetM(float default_m) const;
   uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
   int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
   uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }
   bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }
   bool IsTbTerminal() const { return node_ ? node_->IsTbTerminal() : false; }
   Node::Bounds GetBounds() const; // <<< Return type uses qualified GameResult
   float GetP() const { return edge_ ? edge_->GetP() : 0.0f; }
   lczero::Move GetMove(bool flip = false) const; // <<< Use lczero::Move
   float GetU(float numerator) const;
   std::string DebugString() const;

 protected:
   Edge* edge_ = nullptr;
   Node* node_ = nullptr;
};

// --- Edge_Iterator ---
template <bool is_const>
class Edge_Iterator : public EdgeAndNode {
 public:
  using Ptr = std::conditional_t<is_const, const std::unique_ptr<Node>*,
                                 std::unique_ptr<Node>*>;
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>;
  using value_type = Edge_Iterator;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = Edge_Iterator*;
  using reference = Edge_Iterator&;

  Edge_Iterator();
  Edge_Iterator(NodePtr parent_node);

  Edge_Iterator<is_const> begin();
  Edge_Iterator<is_const> end();
  void operator++();
  Edge_Iterator& operator*();
  Node* GetOrSpawnNode(Node* parent);

 private:
  void Actualize();

  NodePtr parent_node_ = nullptr;
  Ptr node_ptr_ = nullptr;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
};

// --- VisitedNode_Iterator ---
template <bool is_const>
class VisitedNode_Iterator {
 public:
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>;

  VisitedNode_Iterator();
  VisitedNode_Iterator(NodePtr parent_node);

  bool operator==(const VisitedNode_Iterator<is_const>& other) const;
  bool operator!=(const VisitedNode_Iterator<is_const>& other) const { return !(*this == other); }
  VisitedNode_Iterator<is_const> begin();
  VisitedNode_Iterator<is_const> end();
  void operator++();
  Node* operator*();

 private:
   NodePtr parent_node_ = nullptr;
   bool solid_ = false;
   uint16_t total_count_ = 0;
   Node* node_ptr_ = nullptr;
   uint16_t current_idx_ = 0;
};

// Inline definitions
inline Node::ConstIterator Node::Edges() const { return ConstIterator(this); }
inline Node::Iterator Node::Edges() { return Iterator(this); }
inline VisitedNode_Iterator<true> Node::VisitedNodes() const { return VisitedNode_Iterator<true>(this); }
inline VisitedNode_Iterator<false> Node::VisitedNodes() { return VisitedNode_Iterator<false>(this); }

template <bool is_const> inline Edge_Iterator<is_const> Edge_Iterator<is_const>::begin() { return *this; }
template <bool is_const> inline Edge_Iterator<is_const> Edge_Iterator<is_const>::end() { return {}; }
template <bool is_const> inline Edge_Iterator<is_const>& Edge_Iterator<is_const>::operator*() { return *this; }

template <bool is_const> inline VisitedNode_Iterator<is_const> VisitedNode_Iterator<is_const>::begin() { return *this; }
template <bool is_const> inline VisitedNode_Iterator<is_const> VisitedNode_Iterator<is_const>::end() { return {}; }
template <bool is_const> inline Node* VisitedNode_Iterator<is_const>::operator*() {
    if (solid_) { return (node_ptr_ != nullptr && current_idx_ < total_count_) ? &(node_ptr_[current_idx_]) : nullptr; }
    else { return node_ptr_; }
}

// --- NodeTree ---
class NodeTree {
 public:
  ~NodeTree();
  void MakeMove(lczero::Move move); // <<< Use lczero::
  void TrimTreeAtHead();
  bool ResetToPosition(const std::string& starting_fen,
                       const std::vector<std::string>& moves);
  bool ResetToPosition(const lczero::GameState& pos); // <<< Use lczero::
  const lczero::Position& HeadPosition() const; // <<< Use lczero::
  int GetPlyCount() const;
  bool IsBlackToMove() const;
  Node* GetCurrentHead() const;
  Node* GetGameBeginNode() const;
  const lczero::PositionHistory& GetPositionHistory() const; // <<< Use lczero::

 private:
  void DeallocateTree();
  Node* current_head_ = nullptr;
  std::unique_ptr<Node> gamebegin_node_;
  lczero::PositionHistory history_; // <<< Use lczero::
};

}  // namespace classic
}  // namespace lczero
