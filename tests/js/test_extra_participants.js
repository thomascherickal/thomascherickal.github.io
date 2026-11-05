/**
 * test_extra_participants.js - Business Extra Participant Scaling Tests (+ $250/seat scaled by tier)
 * Strict Repo Compliance: Zero emojis, zero em-dashes.
 */

function run(harness, PricingCalculator, constants) {
  const { assert, setSection } = harness;
  const B_IND_PRICE = constants ? constants.B_IND_PRICE : 300;
  const B_EXTRA_SEAT = constants ? constants.B_EXTRA_SEAT : 300;
  const B_FLAG_PRICE = constants ? constants.B_FLAG_PRICE : 3000;

  setSection("Business Extra Participant Scaling Tests");

  const bizT1Extra2Rate = 2 * B_EXTRA_SEAT;
  const bizT1Extra2Expected = B_IND_PRICE + bizT1Extra2Rate;
  const bizT1Extra2 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 12
  });
  assert(
    `Business 12 seats (2 extra @ $${B_EXTRA_SEAT}) @ Tier 1 = $${bizT1Extra2Expected} USD ($${B_IND_PRICE} base + $${bizT1Extra2Rate} extra)`,
    bizT1Extra2.finalPrice === bizT1Extra2Expected && bizT1Extra2.extraParticipants === 2,
    `Expected $${bizT1Extra2Expected}, got $${bizT1Extra2.finalPrice}`
  );

  const bizT2Extra2Base = B_IND_PRICE + (2 * B_EXTRA_SEAT);
  const bizT2Extra2Expected = Math.round(bizT2Extra2Base * 0.75);
  const bizT2Extra2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 12
  });
  assert(
    `Business 12 seats @ Tier 2 (25% parity scaled) = $${bizT2Extra2Expected} USD ($${Math.round(B_IND_PRICE * 0.75)} base + $${Math.round(bizT1Extra2Rate * 0.75)} extra)`,
    bizT2Extra2.finalPrice === bizT2Extra2Expected,
    `Expected $${bizT2Extra2Expected}, got $${bizT2Extra2.finalPrice}`
  );

  const bizT3Extra5Rate = 5 * B_EXTRA_SEAT;
  const bizT3Extra5Base = B_IND_PRICE + bizT3Extra5Rate;
  const bizT3Extra5Expected = Math.round(bizT3Extra5Base * 0.50);
  const bizT3Extra5 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 15
  });
  assert(
    `Business 15 seats @ Tier 3 = $${bizT3Extra5Expected} USD ($${Math.round(B_IND_PRICE * 0.5)} base + $${Math.round(bizT3Extra5Rate * 0.5)} extra)`,
    bizT3Extra5.finalPrice === bizT3Extra5Expected && bizT3Extra5.extraParticipants === 5,
    `Expected $${bizT3Extra5Expected}, got $${bizT3Extra5.finalPrice}`
  );

  const bizT5Extra5Expected = Math.round(bizT3Extra5Base * 0.25);
  const bizT5Extra5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 15
  });
  assert(
    `Business 15 seats @ Tier 5 = $${bizT5Extra5Expected} USD ($${Math.round(B_IND_PRICE * 0.25)} base + $${Math.round(bizT3Extra5Rate * 0.25)} extra)`,
    bizT5Extra5.finalPrice === bizT5Extra5Expected,
    `Expected $${bizT5Extra5Expected}, got $${bizT5Extra5.finalPrice}`
  );

  const biz35Base = B_FLAG_PRICE + (25 * B_EXTRA_SEAT);
  const biz35Expected = biz35Base;
  const biz35Seats = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 35
  });
  assert(
    `Business Flagship 35 seats @ Tier 1 = $${biz35Expected.toLocaleString()} USD ($${B_FLAG_PRICE.toLocaleString()} base + $${(25 * B_EXTRA_SEAT).toLocaleString()} extra)`,
    biz35Seats.finalPrice === biz35Expected && biz35Seats.extraParticipants === 25,
    `Expected $${biz35Expected} with 25 extra, got $${biz35Seats.finalPrice}`
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
  harness.printSummary("JAVASCRIPT UNIT TEST (test_extra_participants.js)", startTime);
  if (harness.results.failed > 0) {
    harness.handleFailures("test_extra_participants");
  }
}

