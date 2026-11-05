/**
 * test_data_integrity.js - Data Integrity & Metadata Tests
 * Strict Repo Compliance: Zero emojis, zero em-dashes.
 */

function run(harness, PricingCalculator) {
  const { assert, setSection } = harness;
  setSection("Data Integrity & Metadata Tests");

  assert(
    "PARITY_TIERS metadata contains all 6 tiers",
    Object.keys(PricingCalculator.PARITY_TIERS).length === 6,
    `Expected 6 tiers, found ${Object.keys(PricingCalculator.PARITY_TIERS).length}`
  );

  assert(
    "PARITY_TIERS multipliers and discounts match defined standards",
    PricingCalculator.PARITY_TIERS[1].multiplier === 1.0 && PricingCalculator.PARITY_TIERS[1].discountPercent === 0 &&
    PricingCalculator.PARITY_TIERS[2].multiplier === 0.75 && PricingCalculator.PARITY_TIERS[2].discountPercent === 25 &&
    PricingCalculator.PARITY_TIERS[3].multiplier === 0.50 && PricingCalculator.PARITY_TIERS[3].discountPercent === 50 &&
    PricingCalculator.PARITY_TIERS[4].multiplier === 0.50 && PricingCalculator.PARITY_TIERS[4].discountPercent === 50 &&
    PricingCalculator.PARITY_TIERS[5].multiplier === 0.25 && PricingCalculator.PARITY_TIERS[5].discountPercent === 75 &&
    PricingCalculator.PARITY_TIERS[6].multiplier === 0.10 && PricingCalculator.PARITY_TIERS[6].discountPercent === 90,
    "Tier multipliers or discounts do not match expected standard"
  );

  assert(
    "Tier 6 is marked isCustom = true",
    PricingCalculator.PARITY_TIERS[6].isCustom === true && PricingCalculator.PARITY_TIERS[1].isCustom === false,
    "Tier 6 should be marked isCustom"
  );

  assert(
    "ALL_SERVICES_DATA contains all 12 services",
    PricingCalculator.ALL_SERVICES_DATA.length === 12,
    `Expected 12 services, found ${PricingCalculator.ALL_SERVICES_DATA.length}`
  );

  const uniqueSlugs = new Set(PricingCalculator.ALL_SERVICES_DATA.map(s => s.slug));
  assert(
    "ALL_SERVICES_DATA has 12 unique slugs without duplicates",
    uniqueSlugs.size === 12,
    `Expected 12 unique slugs, found ${uniqueSlugs.size}`
  );
}

module.exports = { run };

if (typeof require !== 'undefined' && require.main === module) {
  const { createHarness } = require('./test_helpers.js');
  const PricingCalculator = require('../../pricing-calculator.js');
  const harness = createHarness();
  const startTime = Date.now();
  run(harness, PricingCalculator);
  harness.printSummary("JAVASCRIPT UNIT TEST (test_data_integrity.js)", startTime);
  if (harness.results.failed > 0) {
    harness.handleFailures("test_data_integrity");
  }
}

