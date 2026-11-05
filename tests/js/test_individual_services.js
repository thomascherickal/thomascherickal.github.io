/**
 * test_individual_services.js - Individual Pick-Services Tests (strictly 1 person)
 * Strict Repo Compliance: Zero emojis, zero em-dashes.
 */

function run(harness, PricingCalculator, constants) {
  const { assert, setSection } = harness;
  const B_IND_PRICE = constants ? constants.B_IND_PRICE : 300;
  const B_FLAG_PRICE = constants ? constants.B_FLAG_PRICE : 3000;

  setSection("Individual Pick-Services Tests");

  const indivT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    `Individual Pick 1 service @ Tier 1 = $${B_IND_PRICE} USD (0% off)`,
    indivT1.finalPrice === B_IND_PRICE && indivT1.participants === 1 && indivT1.serviceCount === 1,
    `Expected $${B_IND_PRICE}, got $${indivT1.finalPrice}`
  );

  const indivT2Expected = Math.round(B_IND_PRICE * 0.75);
  const indivT2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    `Individual Pick 1 service @ Tier 2 (25% off) = $${indivT2Expected} USD`,
    indivT2.finalPrice === indivT2Expected && indivT2.totalSavings === (B_IND_PRICE - indivT2Expected),
    `Expected $${indivT2Expected}, got $${indivT2.finalPrice}`
  );

  const indivT3Expected = Math.round(B_IND_PRICE * 0.50);
  const indivT3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    `Individual Pick 1 service @ Tier 3 (50% off) = $${indivT3Expected} USD`,
    indivT3.finalPrice === indivT3Expected,
    `Expected $${indivT3Expected}, got $${indivT3.finalPrice}`
  );

  const indivT4 = PricingCalculator.calculatePricing({
    tierId: 4,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    `Individual Pick 1 service @ Tier 4 (Startups 50% off) = $${indivT3Expected} USD`,
    indivT4.finalPrice === indivT3Expected,
    `Expected $${indivT3Expected}, got $${indivT4.finalPrice}`
  );

  const indivT5Expected = Math.round(B_IND_PRICE * 0.25);
  const indivT5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    `Individual Pick 1 service @ Tier 5 (Academia 75% off) = $${indivT5Expected} USD`,
    indivT5.finalPrice === indivT5Expected,
    `Expected $${indivT5Expected}, got $${indivT5.finalPrice}`
  );

  const indivT6 = PricingCalculator.calculatePricing({
    tierId: 6,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    "Individual Pick 1 service @ Tier 6 marks isCustom = true",
    indivT6.isCustom === true && indivT6.tierId === 6,
    "Tier 6 failed to set isCustom"
  );

  const indiv3T1Expected = 3 * B_IND_PRICE;
  const indiv3T1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["s1", "s2", "s3"],
    participants: 1
  });
  assert(
    `Individual Pick 3 services @ Tier 1 = $${indiv3T1Expected} USD`,
    indiv3T1.finalPrice === indiv3T1Expected && indiv3T1.basePrice === indiv3T1Expected && indiv3T1.serviceCount === 3,
    `Expected $${indiv3T1Expected}, got $${indiv3T1.finalPrice}`
  );

  const indiv5T2Base = 5 * B_IND_PRICE;
  const indiv5T2Expected = Math.round(indiv5T2Base * 0.75);
  const indiv5T2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["s1", "s2", "s3", "s4", "s5"],
    participants: 1
  });
  assert(
    `Individual Pick 5 services @ Tier 2 = $${indiv5T2Expected} USD ($${indiv5T2Base.toLocaleString()} base * 0.75)`,
    indiv5T2.finalPrice === indiv5T2Expected && indiv5T2.basePrice === indiv5T2Base,
    `Expected $${indiv5T2Expected}, got $${indiv5T2.finalPrice}`
  );

  const indiv9 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"],
    participants: 1
  });
  const indiv9Expected = 9 * B_IND_PRICE;
  assert(
    `Individual Pick 9 services ($${indiv9Expected.toLocaleString()} USD) does not trigger flagship nudge`,
    indiv9.flagshipCheaper === false && indiv9.finalPrice === indiv9Expected,
    `Expected flagshipCheaper=false, got ${indiv9.flagshipCheaper}`
  );

  const indiv10 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
    participants: 1
  });
  const indiv10Expected = 10 * B_IND_PRICE;
  assert(
    `Individual Pick 10 services ($${indiv10Expected.toLocaleString()} USD) triggers flagship nudge ($${B_FLAG_PRICE.toLocaleString()} threshold)`,
    indiv10.flagshipCheaper === true && indiv10.finalPrice === indiv10Expected,
    `Expected flagshipCheaper=true at $${indiv10Expected}, got ${indiv10.flagshipCheaper}`
  );

  const indivAll12 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12"],
    participants: 1
  });
  const indivAll12Expected = 12 * B_IND_PRICE;
  assert(
    `Individual Pick all 12 services @ Tier 1 = $${indivAll12Expected.toLocaleString()} USD (triggers flagship nudge)`,
    indivAll12.flagshipCheaper === true && indivAll12.finalPrice === indivAll12Expected,
    `Expected flagshipCheaper=true, got ${indivAll12.flagshipCheaper}`
  );

  assert(
    "Individual client mode always forces participants to 1 regardless of user input",
    PricingCalculator.calculatePricing({
      clientMode: "individual",
      packageType: "pick-services",
      selectedServices: ["s1"],
      participants: 25
    }).participants === 1,
    "Individual client mode permitted >1 participants"
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
  harness.printSummary("JAVASCRIPT UNIT TEST (test_individual_services.js)", startTime);
  if (harness.results.failed > 0) {
    harness.handleFailures("test_individual_services");
  }
}

