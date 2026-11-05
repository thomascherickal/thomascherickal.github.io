/**
 * test_business_services.js - Business Pick-Services Tests (up to 10 seats)
 * Strict Repo Compliance: Zero emojis, zero em-dashes.
 */

function run(harness, PricingCalculator, constants) {
  const { assert, setSection } = harness;
  const B_IND_PRICE = constants ? constants.B_IND_PRICE : 300;
  const B_FLAG_PRICE = constants ? constants.B_FLAG_PRICE : 3000;

  setSection("Business Pick-Services Tests");

  const bizT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 10
  });
  assert(
    `Business Pick 1 service @ Tier 1 (10 seats) = $${B_IND_PRICE} USD`,
    bizT1.finalPrice === B_IND_PRICE && bizT1.extraParticipants === 0 && bizT1.totalSavings === 0,
    `Expected $${B_IND_PRICE} with 0 extra, got $${bizT1.finalPrice}`
  );

  const bizT2Expected = Math.round(B_IND_PRICE * 0.75);
  const bizT2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 10
  });
  assert(
    `Business Pick 1 service @ Tier 2 (25% off) = $${bizT2Expected} USD`,
    bizT2.finalPrice === bizT2Expected && bizT2.totalSavings === (B_IND_PRICE - bizT2Expected),
    `Expected $${bizT2Expected}, got $${bizT2.finalPrice}`
  );

  const bizT3Expected = Math.round(B_IND_PRICE * 0.50);
  const bizT3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 10
  });
  assert(
    `Business Pick 1 service @ Tier 3 (50% off) = $${bizT3Expected} USD`,
    bizT3.finalPrice === bizT3Expected,
    `Expected $${bizT3Expected}, got $${bizT3.finalPrice}`
  );

  const bizT5Expected = Math.round(B_IND_PRICE * 0.25);
  const bizT5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 10
  });
  assert(
    `Business Pick 1 service @ Tier 5 (Academia 75% off) = $${bizT5Expected} USD`,
    bizT5.finalPrice === bizT5Expected,
    `Expected $${bizT5Expected}, got $${bizT5.finalPrice}`
  );

  const biz5T1Expected = 5 * B_IND_PRICE;
  const biz5T1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["s1", "s2", "s3", "s4", "s5"],
    participants: 10
  });
  assert(
    `Business Pick 5 services @ Tier 1 = $${biz5T1Expected.toLocaleString()} USD (flagshipCheaper = false)`,
    biz5T1.finalPrice === biz5T1Expected && biz5T1.flagshipCheaper === false,
    `Expected $${biz5T1Expected} and flagshipCheaper=false, got $${biz5T1.finalPrice}`
  );

  const biz9T1Expected = 9 * B_IND_PRICE;
  const biz9T1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"],
    participants: 10
  });
  assert(
    `Business Pick 9 services @ Tier 1 = $${biz9T1Expected.toLocaleString()} USD (flagshipCheaper = false)`,
    biz9T1.finalPrice === biz9T1Expected && biz9T1.flagshipCheaper === false,
    `Expected $${biz9T1Expected} and flagshipCheaper=false, got $${biz9T1.finalPrice}`
  );

  const biz10T1Expected = 10 * B_IND_PRICE;
  const biz10T1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
    participants: 10
  });
  assert(
    `Business Pick 10 services @ Tier 1 = $${biz10T1Expected.toLocaleString()} USD (triggers flagship nudge)`,
    biz10T1.finalPrice === biz10T1Expected && biz10T1.flagshipCheaper === true,
    `Expected $${biz10T1Expected} and flagshipCheaper=true, got $${biz10T1.finalPrice}`
  );

  const bizAll12T1Expected = 12 * B_IND_PRICE;
  const bizAll12T1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12"],
    participants: 10
  });
  assert(
    `Business Pick all 12 services @ Tier 1 = $${bizAll12T1Expected.toLocaleString()} USD (triggers flagship nudge)`,
    bizAll12T1.finalPrice === bizAll12T1Expected && bizAll12T1.flagshipCheaper === true,
    `Expected $${bizAll12T1Expected} and flagshipCheaper=true, got $${bizAll12T1.finalPrice}`
  );

  const bizAll12T3Expected = Math.round((12 * B_IND_PRICE) * 0.50);
  const bizAll12T3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12"],
    participants: 10
  });
  assert(
    `Business Pick all 12 services @ Tier 3 = $${bizAll12T3Expected.toLocaleString()} USD ($${bizAll12T1Expected.toLocaleString()} base * 0.50)`,
    bizAll12T3.finalPrice === bizAll12T3Expected && bizAll12T3.tierId === 3,
    `Expected $${bizAll12T3Expected}, got $${bizAll12T3.finalPrice}`
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
  harness.printSummary("JAVASCRIPT UNIT TEST (test_business_services.js)", startTime);
  if (harness.results.failed > 0) {
    harness.handleFailures("test_business_services");
  }
}

