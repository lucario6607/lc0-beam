#pragma once

#include <memory> // For std::unique_ptr
#include <string>
#include <vector>

#include "chess/gamestate.h"    // For lczero::GameState, lczero::PositionHistory
#include "search/classic/node.h"  // For lczero::classic::Node, lczero::Move

namespace lczero {
namespace classic {

class NodeTree {
 public:
  NodeTree(); // Default constructor declaration
  ~NodeTree(); // Destructor declaration

  Node* GetCurrentHead() const;
  Node* GetGameBeginNode() const;
  const PositionHistory& GetPositionHistory() const;

  // Makes a move, and sets current head to a child node.
  void MakeMove(Move move);

  // Removes children of a current head.
  void TrimTreeAtHead();

  // Resets tree to a given position. Returns true if the current search tree
  // could be reused.
  bool ResetToPosition(const GameState& pos);
  bool ResetToPosition(const std::string& starting_fen,
                       const std::vector<std::string>& moves);

 private:
  void DeallocateTree();

  std::unique_ptr<Node> gamebegin_node_;
  Node* current_head_ = nullptr;
  PositionHistory history_;
};

}  // namespace classic
}  // namespace lczero
