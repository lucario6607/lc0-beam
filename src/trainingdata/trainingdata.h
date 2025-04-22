/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors
  ... (License header) ...
*/

#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "chess/board.h"
#include "chess/gamestate.h"
#include "chess/move.h"
#include "chess/position.h"
#include "chess/types.h"     // <<< Included for lczero::Eval
#include "utils/bitops.h"

namespace lczero {

class TrainingData {
 public:
  // Maximum planes required by any version of the input format.
  static constexpr int MAX_INPUT_PLANES = 119;
  static constexpr int TRAININGDATA_VERSION = 9;

  TrainingData() {}

  struct Chunk {
    PositionHistory history;
    int16_t policy_indices[MAX_OUTPUT_POLICY];
    float policy_values[MAX_OUTPUT_POLICY];
    uint8_t policy_size;
    GameResult game_result;
    Value root_eval;
    Value expected_value;

    // Additional fields for Version >= 8
    float root_q;
    float root_d;
    uint8_t best_ply; // Ply of best move in search relative to current ply.
  };

  // Populates the chunk from position info, policy map and game result.
  // Policy should already be filtered for legal moves.
  static Chunk CreateChunk(const PositionHistory& history,
                           const std::vector<std::pair<Move, float>>& policy,
                           GameResult game_result, Value root_eval = 0.0,
                           Value expected_value = 0.0, float root_q = 0.0f,
                           float root_d = 0.0f, uint8_t best_ply = 0);

  // Encodes the input planes for the neural net (array must be zero-initialized).
  // Returns number of planes written.
  static int FillInputPlanes(const Chunk& chunk,
                             FillEmptyHistory history_fill_type,
                             float planes[MAX_INPUT_PLANES][squares::SIZE]);
  // Decodes training chunk from stdin/socket into a chunk object.
  static std::optional<Chunk> ReadChunk(std::istream& stream);
  // Encodes training chunk object into bytes suitable for storage/transfer.
  static std::string ChunkToString(const Chunk& chunk);

  // Deprecated functions for direct IO from stdin/socket.
  static std::optional<TrainingData> ReadTrainingData(std::istream& stream);
  std::string ToString() const;
  void FillInputPlanes(float planes[MAX_INPUT_PLANES][squares::SIZE]) const;

  // Member variables (using types assumed to be in lczero namespace via includes)
  PositionHistory history;
  int16_t policy_indices[MAX_OUTPUT_POLICY];
  float policy_values[MAX_OUTPUT_POLICY];
  uint8_t policy_size;
  GameResult game_result;
  Value root_eval; // Added in version 7. Use result before that.

  // Added in version 8
  float root_q;
  float root_d;
  uint8_t best_ply; // Ply of best move in search relative to current ply.

  // Added in version 9
  Value expected_value;

 private:
  static std::optional<TrainingData> ReadTrainingDataV6(std::istream& stream);
  static std::optional<TrainingData> ReadTrainingDataV7(std::istream& stream);
  static std::optional<TrainingData> ReadTrainingDataV8(std::istream& stream);
  static std::optional<Chunk> ReadTrainingDataV9(std::istream& stream);
};

// Interface to allow writing training data to different locations.
class TrainingDataWriter {
 public:
  virtual ~TrainingDataWriter() = default;

  // Add new training data chunk.
  virtual void AddChunk(const TrainingData::Chunk& chunk) = 0;
  // Signals end of game. Allows writer to finalize the game etc.
  virtual void GameFinished() = 0;
  // Returns total number of training positions stored.
  virtual std::int64_t GetPositionCount() = 0;
};

// Stores TrainingData::Chunks into specified file.
class TrainingDataFileWriter : public TrainingDataWriter {
 public:
  TrainingDataFileWriter(const std::string& filename);
  ~TrainingDataFileWriter();

  void AddChunk(const TrainingData::Chunk& chunk) override;
  void GameFinished() override {}
  std::int64_t GetPositionCount() override;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace lczero
