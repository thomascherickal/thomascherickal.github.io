/**
 * pricing-tests.js
 * Unit and Functional Test Suite for Pricing Calculator & Dynamic Tier Variations
 * Runs under Node.js or browser console.
 */

const PricingCalculator = typeof require !== 'undefined'
  ? require('./pricing-calculator.js')
  : window.PricingCalculator;

function runTestSuite() {
  const results = {
    total: 0,
    passed: 0,
    failed: 0,
    failures: []
  };

  function assert(testName, condition, message) {
    results.total++;
    if (condition) {
      results.passed++;
      console.log(`[PASS] ${testName}`);
    } else {
      results.failed++;
      console.error(`[FAIL] ${testName}: ${message}`);
      results.failures.push({ testName, message });
    }
  }

  console.log("=================================================");
  console.log("STARTING PRICING CALCULATOR UNIT & FUNCTIONAL TESTS");
  console.log("=================================================");

  // 1. Data Integrity Tests
  assert(
    "PARITY_TIERS metadata contains all 6 tiers",
    Object.keys(PricingCalculator.PARITY_TIERS).length === 6,
    `Expected 6 tiers, found ${Object.keys(PricingCalculator.PARITY_TIERS).length}`
  );

  assert(
    "ALL_SERVICES_DATA contains all 16 services",
    PricingCalculator.ALL_SERVICES_DATA.length === 16,
    `Expected 16 services, found ${PricingCalculator.ALL_SERVICES_DATA.length}`
  );

  // 2. Individual Pick Unit Tests ($1,000 base, strictly 1 person)
  const indivT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    "Individual Pick 1 service @ Tier 1 = $1,000 USD",
    indivT1.finalPrice === 1000 && indivT1.participants === 1,
    `Expected $1000 with 1 participant, got $${indivT1.finalPrice} with ${indivT1.participants}`
  );

  const indivT2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    "Individual Pick 1 service @ Tier 2 (25% off) = $750 USD",
    indivT2.finalPrice === 750 && indivT2.totalSavings === 250,
    `Expected $750 price and $250 savings, got price $${indivT2.finalPrice} and savings $${indivT2.totalSavings}`
  );

  const indivT3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    "Individual Pick 1 service @ Tier 3 (50% off) = $500 USD",
    indivT3.finalPrice === 500 && indivT3.totalSavings === 500,
    `Expected $500 price and $500 savings, got price $${indivT3.finalPrice}`
  );

  const indivT5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    "Individual Pick 1 service @ Tier 5 (75% off) = $250 USD",
    indivT5.finalPrice === 250 && indivT5.totalSavings === 750,
    `Expected $250 price and $750 savings, got price $${indivT5.finalPrice}`
  );

  // 3. Multi-Service Selection Tests (Individuals)
  const indivMulti3 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: [
      "service-generative-ai-transformation",
      "service-local-llms-cost-slashing",
      "service-workflow-automations"
    ],
    participants: 1
  });
  assert(
    "Individual Pick 3 services @ Tier 1 = $3,000 USD",
    indivMulti3.finalPrice === 3000 && indivMulti3.serviceCount === 3,
    `Expected $3000 for 3 services, got $${indivMulti3.finalPrice}`
  );

  const indivAll16 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.map(s => s.slug),
    participants: 1
  });
  assert(
    "Individual Pick all 16 services @ Tier 1 = $16,000 USD",
    indivAll16.finalPrice === 16000 && indivAll16.serviceCount === 16,
    `Expected $16,000, got $${indivAll16.finalPrice}`
  );

  // 4. Business Pick Unit Tests ($2,000 base, up to 20 participants, +$250/extra)
  const bizT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 20
  });
  assert(
    "Business Pick 1 service @ Tier 1 (20 seats) = $2,000 USD",
    bizT1.finalPrice === 2000 && bizT1.extraParticipants === 0,
    `Expected $2,000 with 0 extra, got $${bizT1.finalPrice}`
  );

  const bizT1Extra = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 22
  });
  assert(
    "Business Pick 1 service @ Tier 1 (22 seats: 20 incl + 2 extra @ $250) = $2,500 USD",
    bizT1Extra.finalPrice === 2500 && bizT1Extra.extraHeadRate === 500,
    `Expected $2,500 with $500 extra, got $${bizT1Extra.finalPrice}`
  );

  const bizT2Extra = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 22 // 2 extra heads * $250 * 0.75 = $375; base = $2000 * 0.75 = $1500; total = $1875
  });
  assert(
    "Business Pick 1 service @ Tier 2 (22 seats parity scaled) = $1,875 USD",
    bizT2Extra.finalPrice === 1875 && bizT2Extra.extraHeadRate === 375,
    `Expected $1875 with $375 extra, got $${bizT2Extra.finalPrice}`
  );

  // 5. Flagship Package Calculations
  const flagshipBiz = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 20
  });
  assert(
    "Business Flagship @ Tier 1 = $5,000 USD",
    flagshipBiz.finalPrice === 5000 && flagshipBiz.basePrice === 5000,
    `Expected $5000, got $${flagshipBiz.finalPrice}`
  );

  const flagshipPersonal = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pkg-personal",
    participants: 1
  });
  assert(
    "Personal Flagship @ Tier 1 = $2,500 USD",
    flagshipPersonal.finalPrice === 2500 && flagshipPersonal.basePrice === 2500,
    `Expected $2500, got $${flagshipPersonal.finalPrice}`
  );

  // 6. Functional / Invariant Checks
  assert(
    "Individual client mode always forces participants to 1 regardless of input",
    PricingCalculator.calculatePricing({
      tierId: 1,
      clientMode: "individual",
      packageType: "pick-services",
      selectedServices: ["service-generative-ai-transformation"],
      participants: 50
    }).participants === 1,
    "Individual client mode permitted >1 participants"
  );

  assert(
    "Tier 6 is marked isCustom = true",
    PricingCalculator.PARITY_TIERS[6].isCustom === true,
    "Tier 6 should be marked isCustom"
  );

  // 7. Flagship Cheaper Nudge & Tier 6 Custom Quote Tests (Phase 2)
  const biz2Picks = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: [
      "service-generative-ai-transformation",
      "service-local-llms-cost-slashing"
    ],
    participants: 20
  });
  assert(
    "2 business picks does not trigger flagship nudge (flagshipCheaper = false)",
    biz2Picks.flagshipCheaper === false,
    `Expected flagshipCheaper === false, got ${biz2Picks.flagshipCheaper}`
  );

  const biz3Picks = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: [
      "service-generative-ai-transformation",
      "service-local-llms-cost-slashing",
      "service-workflow-automations"
    ],
    participants: 20
  });
  assert(
    "3 business picks triggers flagship nudge with flagshipPrice 5000",
    biz3Picks.flagshipCheaper === true && biz3Picks.flagshipPrice === 5000,
    `Expected flagshipCheaper === true and flagshipPrice === 5000, got cheaper=${biz3Picks.flagshipCheaper}, price=${biz3Picks.flagshipPrice}`
  );

  const indiv3Picks = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: [
      "service-generative-ai-transformation",
      "service-local-llms-cost-slashing",
      "service-workflow-automations"
    ],
    participants: 1
  });
  assert(
    "3 individual picks triggers flagship nudge with flagshipPrice 2500",
    indiv3Picks.flagshipCheaper === true && indiv3Picks.flagshipPrice === 2500,
    `Expected flagshipCheaper === true and flagshipPrice === 2500, got cheaper=${indiv3Picks.flagshipCheaper}, price=${indiv3Picks.flagshipPrice}`
  );

  const tier6Calc = PricingCalculator.calculatePricing({
    tierId: 6,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 20
  });
  assert(
    "Tier 6 calculation returns isCustom = true",
    tier6Calc.isCustom === true,
    `Expected isCustom === true, got ${tier6Calc.isCustom}`
  );

  console.log("=================================================");
  console.log(`TEST RUN FINISHED: ${results.passed}/${results.total} PASSED`);
  if (results.failed > 0) {
    console.error(`FAILED: ${results.failed} tests`);
    if (typeof process !== 'undefined') process.exit(1);
  } else {
    console.log("ALL TESTS COMPLETED SUCCESSFULLY WITH ZERO ERRORS.");
  }

  return results;
}

if (typeof require !== 'undefined' && require.main === module) {
  runTestSuite();
}

if (typeof window !== 'undefined') {
  window.runPricingTestSuite = runTestSuite;
}
