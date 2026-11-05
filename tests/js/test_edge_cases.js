/**
 * test_edge_cases.js - Defensive Edge Cases & Malformed Inputs Tests
 * Strict Repo Compliance: Zero emojis, zero em-dashes.
 */

function run(harness, PricingCalculator) {
  const { assert, setSection } = harness;
  setSection("Defensive Edge Cases & Malformed Inputs Tests");

  assert(
    "Null/undefined parameter does not throw and defaults to safe calculation",
    PricingCalculator.calculatePricing(null).finalPrice >= 0,
    "calculatePricing(null) threw or failed"
  );

  assert(
    "Invalid tierId defaults safely to Tier 1 (1.0x standard rate)",
    PricingCalculator.calculatePricing({ tierId: 99 }).tierId === 1 && PricingCalculator.calculatePricing({ tierId: -1 }).tierId === 1,
    "Invalid tierId was not defaulted to Tier 1"
  );

  assert(
    "NaN or negative participants defaults to 10 for business and 1 for individual",
    PricingCalculator.calculatePricing({ clientMode: "business", participants: "invalid" }).participants === 10 &&
    PricingCalculator.calculatePricing({ clientMode: "individual", participants: "invalid" }).participants === 1,
    "Malformed participants were not safely normalized"
  );
}

module.exports = { run };

if (typeof require !== 'undefined' && require.main === module) {
  const { createHarness } = require('./test_helpers.js');
  const PricingCalculator = require('../../pricing-calculator.js');
  const harness = createHarness();
  const startTime = Date.now();
  run(harness, PricingCalculator);
  harness.printSummary("JAVASCRIPT UNIT TEST (test_edge_cases.js)", startTime);
  if (harness.results.failed > 0) {
    harness.handleFailures("test_edge_cases");
  }
}

