/**
 * test_custom_package.js - Custom Amount Package Tests
 * Strict Repo Compliance: Zero emojis, zero em-dashes.
 */

function run(harness, PricingCalculator) {
  const { assert, setSection } = harness;
  setSection("Custom Amount Package Tests");

  const custom5000T1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "custom",
    customBase: 5000,
    participants: 10
  });
  assert(
    "Custom scope $5,000 @ Tier 1 = $5,000 USD",
    custom5000T1.finalPrice === 5000 && custom5000T1.basePrice === 5000,
    `Expected $5,000, got $${custom5000T1.finalPrice}`
  );

  const custom5000T2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "business",
    packageType: "custom",
    customBase: 5000,
    participants: 10
  });
  assert(
    "Custom scope $5,000 @ Tier 2 (25% off) = $3,750 USD",
    custom5000T2.finalPrice === 3750 && custom5000T2.totalSavings === 1250,
    `Expected $3,750 with $1,250 savings, got $${custom5000T2.finalPrice}`
  );

  const custom5000T5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "business",
    packageType: "custom",
    customBase: 5000,
    participants: 10
  });
  assert(
    "Custom scope $5,000 @ Tier 5 (75% off) = $1,250 USD",
    custom5000T5.finalPrice === 1250 && custom5000T5.totalSavings === 3750,
    `Expected $1,250 with $3,750 savings, got $${custom5000T5.finalPrice}`
  );

  const customZero = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "custom",
    customBase: 0,
    participants: 10
  });
  assert(
    "Custom scope $0 = $0 USD",
    customZero.finalPrice === 0 && customZero.basePrice === 0,
    `Expected $0, got $${customZero.finalPrice}`
  );

  const customNegative = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "custom",
    customBase: -500,
    participants: 10
  });
  assert(
    "Custom scope negative amount is clamped to $0 USD",
    customNegative.finalPrice === 0 && customNegative.basePrice === 0,
    `Expected $0, got $${customNegative.finalPrice}`
  );
}

module.exports = { run };

if (typeof require !== 'undefined' && require.main === module) {
  const { createHarness } = require('./test_helpers.js');
  const PricingCalculator = require('../../pricing-calculator.js');
  const harness = createHarness();
  const startTime = Date.now();
  run(harness, PricingCalculator);
  harness.printSummary("JAVASCRIPT UNIT TEST (test_custom_package.js)", startTime);
  if (harness.results.failed > 0) {
    harness.handleFailures("test_custom_package");
  }
}

