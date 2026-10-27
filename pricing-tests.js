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

  // 2. Individual Pick Unit Tests ($100 base, strictly 1 person)
  const indivT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    "Individual Pick 1 service @ Tier 1 = $100 USD",
    indivT1.finalPrice === 100 && indivT1.participants === 1,
    `Expected $100 with 1 participant, got $${indivT1.finalPrice} with ${indivT1.participants}`
  );

  const indivT2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    "Individual Pick 1 service @ Tier 2 (25% off) = $75 USD",
    indivT2.finalPrice === 75 && indivT2.totalSavings === 25,
    `Expected $75 price and $25 savings, got price $${indivT2.finalPrice} and savings $${indivT2.totalSavings}`
  );

  const indivT3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    "Individual Pick 1 service @ Tier 3 (50% off) = $50 USD",
    indivT3.finalPrice === 50 && indivT3.totalSavings === 50,
    `Expected $50 price and $50 savings, got price $${indivT3.finalPrice}`
  );

  const indivT5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    "Individual Pick 1 service @ Tier 5 (75% off) = $25 USD",
    indivT5.finalPrice === 25 && indivT5.totalSavings === 75,
    `Expected $25 price and $75 savings, got price $${indivT5.finalPrice}`
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
    "Individual Pick 3 services @ Tier 1 = $300 USD",
    indivMulti3.finalPrice === 300 && indivMulti3.serviceCount === 3,
    `Expected $300 for 3 services, got $${indivMulti3.finalPrice}`
  );

  const indivAll16 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.map(s => s.slug),
    participants: 1
  });
  assert(
    "Individual Pick all 16 services @ Tier 1 = $1,600 USD",
    indivAll16.finalPrice === 1600 && indivAll16.serviceCount === 16,
    `Expected $1,600, got $${indivAll16.finalPrice}`
  );

  // 4. Business Pick Unit Tests ($200 base, up to 20 participants, +$100/extra)
  const bizT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 20
  });
  assert(
    "Business Pick 1 service @ Tier 1 (20 seats) = $200 USD",
    bizT1.finalPrice === 200 && bizT1.extraParticipants === 0,
    `Expected $200 with 0 extra, got $${bizT1.finalPrice}`
  );

  const bizT1Extra = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 22
  });
  assert(
    "Business Pick 1 service @ Tier 1 (22 seats: 20 incl + 2 extra @ $100) = $400 USD",
    bizT1Extra.finalPrice === 400 && bizT1Extra.extraHeadRate === 200,
    `Expected $400 with $200 extra, got $${bizT1Extra.finalPrice}`
  );

  const bizT2Extra = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 22 // 2 extra heads * $100 * 0.75 = $150; base = $200 * 0.75 = $150; total = $300
  });
  assert(
    "Business Pick 1 service @ Tier 2 (22 seats parity scaled) = $300 USD",
    bizT2Extra.finalPrice === 300 && bizT2Extra.extraHeadRate === 150,
    `Expected $300 with $150 extra, got $${bizT2Extra.finalPrice}`
  );

  // 5. Flagship Package Calculations
  const flagshipBiz = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 20
  });
  assert(
    "Business Flagship @ Tier 1 = $2,000 USD",
    flagshipBiz.finalPrice === 2000 && flagshipBiz.basePrice === 2000,
    `Expected $2000, got $${flagshipBiz.finalPrice}`
  );

  const flagshipPersonal = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pkg-personal",
    participants: 1
  });
  assert(
    "Personal Flagship @ Tier 1 = $1,000 USD",
    flagshipPersonal.finalPrice === 1000 && flagshipPersonal.basePrice === 1000,
    `Expected $1000, got $${flagshipPersonal.finalPrice}`
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

  const biz10Picks = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 10).map(s => s.slug),
    participants: 20
  });
  assert(
    "10 business picks triggers flagship nudge with flagshipPrice 2000",
    biz10Picks.flagshipCheaper === true && biz10Picks.flagshipPrice === 2000,
    `Expected flagshipCheaper === true and flagshipPrice === 2000, got cheaper=${biz10Picks.flagshipCheaper}, price=${biz10Picks.flagshipPrice}`
  );

  const indiv10Picks = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 10).map(s => s.slug),
    participants: 1
  });
  assert(
    "10 individual picks triggers flagship nudge with flagshipPrice 1000",
    indiv10Picks.flagshipCheaper === true && indiv10Picks.flagshipPrice === 1000,
    `Expected flagshipCheaper === true and flagshipPrice === 1000, got cheaper=${indiv10Picks.flagshipCheaper}, price=${indiv10Picks.flagshipPrice}`
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
