/**
 * test_flagship_packages.js - Flagship Package Tests
 * Strict Repo Compliance: Zero emojis, zero em-dashes.
 */

function run(harness, PricingCalculator, constants) {
  const { assert, setSection } = harness;
  const B_FLAG_PRICE = constants ? constants.B_FLAG_PRICE : 3000;
  const P_FLAG_PRICE = constants ? constants.P_FLAG_PRICE : 3000;

  setSection("Flagship Package Tests");

  const flagshipBizT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 10
  });
  assert(
    `Business Flagship @ Tier 1 = $${B_FLAG_PRICE.toLocaleString()} USD (0% off)`,
    flagshipBizT1.finalPrice === B_FLAG_PRICE && flagshipBizT1.basePrice === B_FLAG_PRICE && flagshipBizT1.totalSavings === 0,
    `Expected $${B_FLAG_PRICE}, got $${flagshipBizT1.finalPrice}`
  );

  const flagshipBizT2Expected = Math.round(B_FLAG_PRICE * 0.75);
  const flagshipBizT2Savings = B_FLAG_PRICE - flagshipBizT2Expected;
  const flagshipBizT2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 10
  });
  assert(
    `Business Flagship @ Tier 2 = $${flagshipBizT2Expected.toLocaleString()} USD (25% off, $${flagshipBizT2Savings.toLocaleString()} savings)`,
    flagshipBizT2.finalPrice === flagshipBizT2Expected && flagshipBizT2.totalSavings === flagshipBizT2Savings,
    `Expected $${flagshipBizT2Expected} with $${flagshipBizT2Savings} savings, got $${flagshipBizT2.finalPrice}`
  );

  const flagshipBizT3Expected = Math.round(B_FLAG_PRICE * 0.50);
  const flagshipBizT3Savings = B_FLAG_PRICE - flagshipBizT3Expected;
  const flagshipBizT3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 10
  });
  assert(
    `Business Flagship @ Tier 3 = $${flagshipBizT3Expected.toLocaleString()} USD (50% off, $${flagshipBizT3Savings.toLocaleString()} savings)`,
    flagshipBizT3.finalPrice === flagshipBizT3Expected && flagshipBizT3.totalSavings === flagshipBizT3Savings,
    `Expected $${flagshipBizT3Expected} with $${flagshipBizT3Savings} savings, got $${flagshipBizT3.finalPrice}`
  );

  const flagshipBizT5Expected = Math.round(B_FLAG_PRICE * 0.25);
  const flagshipBizT5Savings = B_FLAG_PRICE - flagshipBizT5Expected;
  const flagshipBizT5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 10
  });
  assert(
    `Business Flagship @ Tier 5 = $${flagshipBizT5Expected.toLocaleString()} USD (75% off, $${flagshipBizT5Savings.toLocaleString()} savings)`,
    flagshipBizT5.finalPrice === flagshipBizT5Expected && flagshipBizT5.totalSavings === flagshipBizT5Savings,
    `Expected $${flagshipBizT5Expected} with $${flagshipBizT5Savings} savings, got $${flagshipBizT5.finalPrice}`
  );

  const flagshipPersonalT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pkg-personal",
    participants: 1
  });
  assert(
    `Personal Flagship @ Tier 1 = $${P_FLAG_PRICE.toLocaleString()} USD (0% off)`,
    flagshipPersonalT1.finalPrice === P_FLAG_PRICE && flagshipPersonalT1.basePrice === P_FLAG_PRICE && flagshipPersonalT1.totalSavings === 0,
    `Expected $${P_FLAG_PRICE}, got $${flagshipPersonalT1.finalPrice}`
  );

  const flagshipPersonalT2Expected = Math.round(P_FLAG_PRICE * 0.75);
  const flagshipPersonalT2Savings = P_FLAG_PRICE - flagshipPersonalT2Expected;
  const flagshipPersonalT2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "individual",
    packageType: "pkg-personal",
    participants: 1
  });
  assert(
    `Personal Flagship @ Tier 2 = $${flagshipPersonalT2Expected.toLocaleString()} USD (25% off, $${flagshipPersonalT2Savings.toLocaleString()} savings)`,
    flagshipPersonalT2.finalPrice === flagshipPersonalT2Expected && flagshipPersonalT2.totalSavings === flagshipPersonalT2Savings,
    `Expected $${flagshipPersonalT2Expected} with $${flagshipPersonalT2Savings} savings, got $${flagshipPersonalT2.finalPrice}`
  );

  const flagshipPersonalT3Expected = Math.round(P_FLAG_PRICE * 0.50);
  const flagshipPersonalT3Savings = P_FLAG_PRICE - flagshipPersonalT3Expected;
  const flagshipPersonalT3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "individual",
    packageType: "pkg-personal",
    participants: 1
  });
  assert(
    `Personal Flagship @ Tier 3 = $${flagshipPersonalT3Expected.toLocaleString()} USD (50% off, $${flagshipPersonalT3Savings.toLocaleString()} savings)`,
    flagshipPersonalT3.finalPrice === flagshipPersonalT3Expected && flagshipPersonalT3.totalSavings === flagshipPersonalT3Savings,
    `Expected $${flagshipPersonalT3Expected} with $${flagshipPersonalT3Savings} savings, got $${flagshipPersonalT3.finalPrice}`
  );

  const flagshipPersonalT5Expected = Math.round(P_FLAG_PRICE * 0.25);
  const flagshipPersonalT5Savings = P_FLAG_PRICE - flagshipPersonalT5Expected;
  const flagshipPersonalT5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "individual",
    packageType: "pkg-personal",
    participants: 1
  });
  assert(
    `Personal Flagship @ Tier 5 = $${flagshipPersonalT5Expected.toLocaleString()} USD (75% off, $${flagshipPersonalT5Savings.toLocaleString()} savings)`,
    flagshipPersonalT5.finalPrice === flagshipPersonalT5Expected && flagshipPersonalT5.totalSavings === flagshipPersonalT5Savings,
    `Expected $${flagshipPersonalT5Expected} with $${flagshipPersonalT5Savings} savings, got $${flagshipPersonalT5.finalPrice}`
  );
}

module.exports = { run };

if (typeof require !== 'undefined' && require.main === module) {
  const { createHarness } = require('./test_helpers.js');
  const PricingCalculator = require('../../pricing-calculator.js');
  const { constants } = require('../../pricing-tests.js');
  const harness = createHarness();
  const startTime = Date.now();
  run(harness, PricingCalculator, constants);
  harness.printSummary("JAVASCRIPT UNIT TEST (test_flagship_packages.js)", startTime);
  if (harness.results.failed > 0) {
    harness.handleFailures("test_flagship_packages");
  }
}

