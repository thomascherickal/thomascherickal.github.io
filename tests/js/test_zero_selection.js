/**
 * test_zero_selection.js - Nothing Selected (0 USD) Tests
 * Strict Repo Compliance: Zero emojis, zero em-dashes.
 */

function run(harness, PricingCalculator) {
  const { assert, setSection } = harness;
  setSection("Nothing Selected (0 USD) Tests");

  const zeroIndiv = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: [],
    participants: 1
  });
  assert(
    "Nothing Selected (Individual): Displays 0 USD ($0 final price, 0 base)",
    zeroIndiv.finalPrice === 0 && zeroIndiv.basePrice === 0 && zeroIndiv.serviceCount === 0 && zeroIndiv.totalSavings === 0,
    `Expected finalPrice=0 and basePrice=0, got finalPrice=${zeroIndiv.finalPrice}, basePrice=${zeroIndiv.basePrice}`
  );

  const zeroBiz = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: [],
    participants: 20
  });
  assert(
    "Nothing Selected (Business): Displays 0 USD ($0 final price, 0 base)",
    zeroBiz.finalPrice === 0 && zeroBiz.basePrice === 0 && zeroBiz.serviceCount === 0,
    `Expected finalPrice=0, got ${zeroBiz.finalPrice}`
  );

  const zeroBizExtraHeads = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: [],
    participants: 50 // 30 extra participants entered, but 0 services picked
  });
  assert(
    "Nothing Selected with Extra Participants: Displays 0 USD without extra head charge",
    zeroBizExtraHeads.finalPrice === 0 && zeroBizExtraHeads.extraParticipants === 0,
    `Expected 0 USD finalPrice and 0 extra heads charge, got finalPrice=${zeroBizExtraHeads.finalPrice}, extraParts=${zeroBizExtraHeads.extraParticipants}`
  );

  [1, 2, 3, 4, 5, 6].forEach(tier => {
    const res = PricingCalculator.calculatePricing({
      tierId: tier,
      clientMode: "business",
      packageType: "pick-services",
      selectedServices: [],
      participants: 10
    });
    assert(
      `Nothing Selected @ Tier ${tier} returns $0 USD`,
      res.finalPrice === 0 && res.basePrice === 0,
      `Tier ${tier} failed: expected $0, got finalPrice=${res.finalPrice}`
    );
  });

  assert(
    "Nothing Selected does not trigger flagship cheaper nudge",
    zeroBiz.flagshipCheaper === false && zeroIndiv.flagshipCheaper === false,
    "flagshipCheaper should be false when 0 services are selected"
  );
}

module.exports = { run };

if (typeof require !== 'undefined' && require.main === module) {
  const { createHarness } = require('./test_helpers.js');
  const PricingCalculator = require('../../pricing-calculator.js');
  const harness = createHarness();
  const startTime = Date.now();
  run(harness, PricingCalculator);
  harness.printSummary("JAVASCRIPT UNIT TEST (test_zero_selection.js)", startTime);
  if (harness.results.failed > 0) {
    harness.handleFailures("test_zero_selection");
  }
}

