// --- START OF FILE search/classic/params.h ---

/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include "neural/encoder.h" // Assuming this path is correct for your version
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"
#include "neural/shared_params.h" // Include shared params

namespace lczero {
namespace classic { // Added classic namespace

enum class ContemptMode { PLAY, WHITE, BLACK, NONE };

class SearchParams {
 public:
  SearchParams(const OptionsDict& options);
  SearchParams(const SearchParams&) = delete;

  // Use struct for WDLRescaleParams calculation to make them const.
  struct WDLRescaleParams {
    WDLRescaleParams(float r, float d) {
      ratio = r;
      diff = d;
    }
    float ratio;
    float diff;
  };

  // Populates UciOptions with search parameters.
  static void Populate(OptionsParser* options);

  // Parameter getters.
  int GetMiniBatchSize() const { return kMiniBatchSize; }
   int GetMaxPrefetchBatch() const {
    return options_.Get<int>(kMaxPrefetchBatchId);
  }
  float GetCpuct(bool at_root) const { return at_root ? kCpuctAtRoot : kCpuct; }
  float GetCpuctBase(bool at_root) const {
    return at_root ? kCpuctBaseAtRoot : kCpuctBase;
  }
   float GetCpuctExponent(bool at_root) const { // Added missing getter
    return at_root ? kCpuctExponentAtRoot : kCpuctExponent;
  }
  float GetCpuctFactor(bool at_root) const {
    return at_root ? kCpuctFactorAtRoot : kCpuctFactor;
  }
  bool GetTwoFoldDraws() const { return kTwoFoldDraws; }
  float GetTemperature() const { return options_.Get<float>(kTemperatureId); }
  float GetTemperatureVisitOffset() const {
    return options_.Get<float>(kTemperatureVisitOffsetId);
  }
  int GetTempDecayMoves() const { return options_.Get<int>(kTempDecayMovesId); }
  int GetTempDecayDelayMoves() const {
    return options_.Get<int>(kTempDecayDelayMovesId);
  }
  int GetTemperatureCutoffMove() const {
    return options_.Get<int>(kTemperatureCutoffMoveId);
  }
  float GetTemperatureEndgame() const {
    return options_.Get<float>(kTemperatureEndgameId);
  }
  float GetTemperatureWinpctCutoff() const {
    return options_.Get<float>(kTemperatureWinpctCutoffId);
  }
  float GetNoiseEpsilon() const { return kNoiseEpsilon; }
  float GetNoiseAlpha() const { return kNoiseAlpha; }
  bool GetVerboseStats() const { return options_.Get<bool>(kVerboseStatsId); }
  bool GetLogLiveStats() const { return options_.Get<bool>(kLogLiveStatsId); }
  bool GetFpuAbsolute(bool at_root) const {
    return at_root ? kFpuAbsoluteAtRoot : kFpuAbsolute;
  }
  float GetFpuValue(bool at_root) const {
    return at_root ? kFpuValueAtRoot : kFpuValue;
  }
  int GetCacheHistoryLength() const { return kCacheHistoryLength; }
  float GetPolicySoftmaxTemp() const { return kPolicySoftmaxTemp; }
  int GetMaxCollisionEvents() const { return kMaxCollisionEvents; }
  int GetMaxCollisionVisits() const { return kMaxCollisionVisits; }
  bool GetOutOfOrderEval() const { return kOutOfOrderEval; }
  bool GetStickyEndgames() const { return kStickyEndgames; }
  bool GetSyzygyFastPlay() const { return kSyzygyFastPlay; }
  int GetMultiPv() const { return options_.Get<int>(kMultiPvId); }
  bool GetPerPvCounters() const { return options_.Get<bool>(kPerPvCountersId); }
  std::string GetScoreType() const {
    return options_.Get<std::string>(kScoreTypeId);
  }
  FillEmptyHistory GetHistoryFill() const { return kHistoryFill; } // Note: Uses shared param getter
  float GetMovesLeftMaxEffect() const { return kMovesLeftMaxEffect; }
  float GetMovesLeftThreshold() const { return kMovesLeftThreshold; }
  float GetMovesLeftSlope() const { return kMovesLeftSlope; }
  float GetMovesLeftConstantFactor() const { return kMovesLeftConstantFactor; }
  float GetMovesLeftScaledFactor() const { return kMovesLeftScaledFactor; }
  float GetMovesLeftQuadraticFactor() const {
    return kMovesLeftQuadraticFactor;
  }
  bool GetDisplayCacheUsage() const { return kDisplayCacheUsage; } // Restored getter
  int GetMaxConcurrentSearchers() const { return kMaxConcurrentSearchers; }
  float GetDrawScore() const { return kDrawScore; }
  ContemptMode GetContemptMode() const {
    std::string mode = options_.Get<std::string>(kContemptModeId);
    if (mode == "play") return ContemptMode::PLAY;
    if (mode == "white_side_analysis") return ContemptMode::WHITE;
    if (mode == "black_side_analysis") return ContemptMode::BLACK;
    assert(mode == "disable");
    return ContemptMode::NONE;
  }
  float GetWDLRescaleRatio() const { return kWDLRescaleParams.ratio; }
  float GetWDLRescaleDiff() const { return kWDLRescaleParams.diff; }
  float GetWDLMaxS() const { return kWDLMaxS; } // Added getter
  float GetWDLEvalObjectivity() const { return kWDLEvalObjectivity; }
  float GetMaxOutOfOrderEvalsFactor() const { // Changed return type and name
    return kMaxOutOfOrderEvalsFactor;
  }
  uint32_t GetMaxOutOfOrderEvals() const { return kMaxOutOfOrderEvals; } // Added derived getter
  float GetNpsLimit() const { return kNpsLimit; }
  int GetSolidTreeThreshold() const { return kSolidTreeThreshold; } // Added getter

  int GetTaskWorkersPerSearchWorker() const {
    return kTaskWorkersPerSearchWorker;
  }
  int GetMinimumWorkSizeForProcessing() const {
    return kMinimumWorkSizeForProcessing;
  }
  int GetMinimumWorkSizeForPicking() const {
    return kMinimumWorkSizeForPicking;
  }
  int GetMinimumRemainingWorkSizeForPicking() const {
    return kMinimumRemainingWorkSizeForPicking;
  }
  int GetMinimumWorkPerTaskForProcessing() const {
    return kMinimumWorkPerTaskForProcessing;
  }
  int GetIdlingMinimumWork() const { return kIdlingMinimumWork; }
  int GetThreadIdlingThreshold() const { return kThreadIdlingThreshold; }
  int GetMaxCollisionVisitsScalingStart() const {
    return kMaxCollisionVisitsScalingStart;
  }
  int GetMaxCollisionVisitsScalingEnd() const {
    return kMaxCollisionVisitsScalingEnd;
  }
  float GetMaxCollisionVisitsScalingPower() const {
    return kMaxCollisionVisitsScalingPower;
  }
  bool GetSearchSpinBackoff() const { return kSearchSpinBackoff; }

  // Search parameter IDs.
  static const OptionId kMiniBatchSizeId;
  static const OptionId kMaxPrefetchBatchId; // Added ID
  static const OptionId kCpuctId;
  static const OptionId kCpuctAtRootId;
  static const OptionId kCpuctExponentId; // Added ID
  static const OptionId kCpuctExponentAtRootId; // Added ID
  static const OptionId kCpuctBaseId;
  static const OptionId kCpuctBaseAtRootId;
  static const OptionId kCpuctFactorId;
  static const OptionId kCpuctFactorAtRootId;
  static const OptionId kRootHasOwnCpuctParamsId;
  static const OptionId kTwoFoldDrawsId;
  static const OptionId kTemperatureId;
  static const OptionId kTempDecayMovesId;
  static const OptionId kTempDecayDelayMovesId;
  static const OptionId kTemperatureCutoffMoveId;
  static const OptionId kTemperatureEndgameId;
  static const OptionId kTemperatureWinpctCutoffId;
  static const OptionId kTemperatureVisitOffsetId;
  static const OptionId kNoiseEpsilonId;
  static const OptionId kNoiseAlphaId;
  static const OptionId kVerboseStatsId;
  static const OptionId kLogLiveStatsId;
  static const OptionId kFpuStrategyId;
  static const OptionId kFpuValueId;
  static const OptionId kFpuStrategyAtRootId;
  static const OptionId kFpuValueAtRootId;
  static const OptionId kCacheHistoryLengthId;
  // static const OptionId kPolicySoftmaxTempId; // Usually comes from shared params
  static const OptionId kMaxCollisionEventsId;
  static const OptionId kMaxCollisionVisitsId;
  static const OptionId kOutOfOrderEvalId;
  static const OptionId kStickyEndgamesId;
  static const OptionId kSyzygyFastPlayId;
  static const OptionId kMultiPvId;
  static const OptionId kPerPvCountersId;
  static const OptionId kScoreTypeId;
  // static const OptionId kHistoryFillId; // Usually comes from shared params
  static const OptionId kMovesLeftMaxEffectId;
  static const OptionId kMovesLeftThresholdId;
  static const OptionId kMovesLeftConstantFactorId;
  static const OptionId kMovesLeftScaledFactorId;
  static const OptionId kMovesLeftQuadraticFactorId;
  static const OptionId kMovesLeftSlopeId;
  static const OptionId kDisplayCacheUsageId; // Restored ID
  static const OptionId kMaxConcurrentSearchersId;
  static const OptionId kDrawScoreId;
  static const OptionId kContemptModeId;
  static const OptionId kContemptId;
  static const OptionId kContemptMaxValueId;
  static const OptionId kWDLCalibrationEloId;
  static const OptionId kWDLContemptAttenuationId;
  static const OptionId kWDLMaxSId; // Added ID
  static const OptionId kWDLEvalObjectivityId;
  static const OptionId kWDLDrawRateTargetId;
  static const OptionId kWDLDrawRateReferenceId;
  static const OptionId kWDLBookExitBiasId;
  static const OptionId kMaxOutOfOrderEvalsFactorId; // Changed name
  static const OptionId kNpsLimitId;
  static const OptionId kSolidTreeThresholdId; // Added ID
  static const OptionId kTaskWorkersPerSearchWorkerId;
  static const OptionId kMinimumWorkSizeForProcessingId;
  static const OptionId kMinimumWorkSizeForPickingId;
  static const OptionId kMinimumRemainingWorkSizeForPickingId;
  static const OptionId kMinimumWorkPerTaskForProcessingId;
  static const OptionId kIdlingMinimumWorkId;
  static const OptionId kThreadIdlingThresholdId;
  static const OptionId kMaxCollisionVisitsScalingStartId;
  static const OptionId kMaxCollisionVisitsScalingEndId;
  static const OptionId kMaxCollisionVisitsScalingPowerId;
  static const OptionId kUCIOpponentId;
  static const OptionId kUCIRatingAdvId;
  static const OptionId kSearchSpinBackoffId;

 private:
  const OptionsDict& options_;
  // Cached parameter values. Values have to be cached if either:
  // 1. Parameter is accessed often and has to be cached for performance reasons.
  // 2. Parameter has to stay the same during the search.
  // TODO(crem) Some of those parameters can be converted to be dynamic after trivial search optimizations.
  const float kCpuct;
  const float kCpuctAtRoot;
  const float kCpuctExponent; // Added member
  const float kCpuctExponentAtRoot; // Added member
  const float kCpuctBase;
  const float kCpuctBaseAtRoot;
  const float kCpuctFactor;
  const float kCpuctFactorAtRoot;
  const bool kTwoFoldDraws;
  const float kNoiseEpsilon;
  const float kNoiseAlpha;
  const bool kFpuAbsolute;
  const float kFpuValue;
  const bool kFpuAbsoluteAtRoot;
  const float kFpuValueAtRoot;
  const int kCacheHistoryLength;
  const float kPolicySoftmaxTemp; // Kept, assuming it might be classic-specific override
  const int kMaxCollisionEvents;
  const int kMaxCollisionVisits;
  const bool kOutOfOrderEval;
  const bool kStickyEndgames;
  const bool kSyzygyFastPlay;
  const FillEmptyHistory kHistoryFill; // Kept, assuming classic-specific override
  const int kMiniBatchSize;
  const float kMovesLeftMaxEffect;
  const float kMovesLeftThreshold;
  const float kMovesLeftSlope;
  const float kMovesLeftConstantFactor;
  const float kMovesLeftScaledFactor;
  const float kMovesLeftQuadraticFactor;
  const bool kDisplayCacheUsage; // Restored member
  const int kMaxConcurrentSearchers;
  const float kDrawScore;
  const float kContempt;
  const WDLRescaleParams kWDLRescaleParams;
  const float kWDLMaxS; // Added member
  const float kWDLEvalObjectivity;
  const float kMaxOutOfOrderEvalsFactor; // Changed type and name
  const uint32_t kMaxOutOfOrderEvals; // Added derived member
  const float kNpsLimit;
  const int kSolidTreeThreshold; // Added member
  const int kTaskWorkersPerSearchWorker;
  const int kMinimumWorkSizeForProcessing;
  const int kMinimumWorkSizeForPicking;
  const int kMinimumRemainingWorkSizeForPicking;
  const int kMinimumWorkPerTaskForProcessing;
  const int kIdlingMinimumWork;
  const int kThreadIdlingThreshold;
  const int kMaxCollisionVisitsScalingStart;
  const int kMaxCollisionVisitsScalingEnd;
  const float kMaxCollisionVisitsScalingPower;
  const bool kSearchSpinBackoff;
  // Removed members related to features not obviously present in the provided header:
  // kCpuctUtilityStdevPrior, kCpuctUtilityStdevScale, kCpuctUtilityStdevPriorWeight,
  // kUseVarianceScaling, kMoveRuleBucketing, kUncertaintyWeightingCap,
  // kUncertaintyWeightingCoefficient, kUncertaintyWeightingExponent,
  // kUseUncertaintyWeighting, kEasyEvalWeightDecay,
  // kCpuctUncertaintyMinFactor, kCpuctUncertaintyMaxFactor,
  // kCpuctUncertaintyMinUncertainty, kCpuctUncertaintyMaxUncertainty,
  // kUseCpuctUncertainty, kJustFpuUncertainty, kTopPolicyBoost,
  // kTopPolicyNumBoost, kTopPolicyTierTwoBoost, kTopPolicyTierTwoNumBoost,
  // kUsePolicyBoosting, kPolicyDecayExponent, kPolicyDecayFactor,
  // kDesperationMultiplier, kDesperationLow, kDesperationHigh,
  // kDesperationPriorWeight, kUseDesperation, kUseCorrectionHistory,
  // kCorrectionHistoryAlpha, kCorrectionHistoryLambda

};

}  // namespace classic
}  // namespace lczero
