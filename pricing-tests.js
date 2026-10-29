/**
 * pricing-tests.js
 * Comprehensive Unit and Functional Test Suite for Pricing Calculator & Dynamic UI Pane
 * Runs under Node.js or browser console with ZERO external dependencies.
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

  // ---------------------------------------------------------------------------
  // SECTION 1: Data Integrity & Metadata Tests
  // ---------------------------------------------------------------------------
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

  // ---------------------------------------------------------------------------
  // SECTION 2: Nothing Selected (0 USD) Tests
  // ---------------------------------------------------------------------------
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
    zeroBizExtraHeads.finalPrice === 0 && zeroBizExtraHeads.extraParticipants === 0 && zeroBizExtraHeads.extraHeadRate === 0,
    `Expected $0 USD, got $${zeroBizExtraHeads.finalPrice} with extraHeadRate=${zeroBizExtraHeads.extraHeadRate}`
  );

  [1, 2, 3, 4, 5, 6].forEach(tier => {
    const calcZeroTier = PricingCalculator.calculatePricing({
      tierId: tier,
      clientMode: "business",
      packageType: "pick-services",
      selectedServices: []
    });
    assert(
      `Nothing Selected @ Tier ${tier} returns $0 USD`,
      calcZeroTier.finalPrice === 0 && calcZeroTier.basePrice === 0,
      `Expected $0 for Tier ${tier}, got $${calcZeroTier.finalPrice}`
    );
  });

  assert(
    "Nothing Selected does not trigger flagship cheaper nudge",
    zeroBiz.flagshipCheaper === false && zeroIndiv.flagshipCheaper === false,
    "flagshipCheaper should be false when 0 services are selected"
  );

  // ---------------------------------------------------------------------------
  // SECTION 3: Individual Pick-Services Tests ($100/service base, strictly 1 person)
  // ---------------------------------------------------------------------------
  const indivT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    "Individual Pick 1 service @ Tier 1 = $100 USD (0% off)",
    indivT1.finalPrice === 100 && indivT1.participants === 1 && indivT1.totalSavings === 0,
    `Expected $100 with 1 participant, got $${indivT1.finalPrice}`
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
    `Expected $75 price and $25 savings, got price $${indivT2.finalPrice}`
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

  const indivT4 = PricingCalculator.calculatePricing({
    tierId: 4,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    "Individual Pick 1 service @ Tier 4 (Startups 50% off) = $50 USD",
    indivT4.finalPrice === 50 && indivT4.totalSavings === 50,
    `Expected $50 price and $50 savings, got price $${indivT4.finalPrice}`
  );

  const indivT5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    "Individual Pick 1 service @ Tier 5 (Academia 75% off) = $25 USD",
    indivT5.finalPrice === 25 && indivT5.totalSavings === 75,
    `Expected $25 price and $75 savings, got price $${indivT5.finalPrice}`
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
    indivT6.isCustom === true && indivT6.tier.id === 6,
    `Expected isCustom=true, got ${indivT6.isCustom}`
  );

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

  const indivMulti5T2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 5).map(s => s.slug),
    participants: 1
  });
  assert(
    "Individual Pick 5 services @ Tier 2 = $375 USD ($500 base * 0.75)",
    indivMulti5T2.finalPrice === 375 && indivMulti5T2.totalSavings === 125,
    `Expected $375 with $125 savings, got $${indivMulti5T2.finalPrice}`
  );

  const indivMulti9 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 9).map(s => s.slug),
    participants: 1
  });
  assert(
    "Individual Pick 9 services ($900 USD) does not trigger flagship nudge",
    indivMulti9.finalPrice === 900 && indivMulti9.flagshipCheaper === false,
    `Expected $900 and flagshipCheaper=false, got price=${indivMulti9.finalPrice}, cheaper=${indivMulti9.flagshipCheaper}`
  );

  const indivMulti10 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 10).map(s => s.slug),
    participants: 1
  });
  assert(
    "Individual Pick 10 services ($1,000 USD) triggers flagship nudge ($1,000 threshold)",
    indivMulti10.finalPrice === 1000 && indivMulti10.flagshipCheaper === true && indivMulti10.flagshipPrice === 1000,
    `Expected $1,000 and flagshipCheaper=true, got price=${indivMulti10.finalPrice}, cheaper=${indivMulti10.flagshipCheaper}`
  );

  const indivAll12 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.map(s => s.slug),
    participants: 1
  });
  assert(
    "Individual Pick all 12 services @ Tier 1 = $1,200 USD (triggers flagship nudge)",
    indivAll12.finalPrice === 1200 && indivAll12.serviceCount === 12 && indivAll12.flagshipCheaper === true,
    `Expected $1,200, got $${indivAll12.finalPrice}`
  );

  assert(
    "Individual client mode always forces participants to 1 regardless of user input",
    PricingCalculator.calculatePricing({
      tierId: 1,
      clientMode: "individual",
      packageType: "pick-services",
      selectedServices: ["service-generative-ai-transformation"],
      participants: 50
    }).participants === 1,
    "Individual client mode permitted >1 participants"
  );

  // ---------------------------------------------------------------------------
  // SECTION 4: Business Pick-Services Tests ($200/service base, up to 20 seats)
  // ---------------------------------------------------------------------------
  const bizT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 20
  });
  assert(
    "Business Pick 1 service @ Tier 1 (20 seats) = $200 USD",
    bizT1.finalPrice === 200 && bizT1.extraParticipants === 0 && bizT1.totalSavings === 0,
    `Expected $200 with 0 extra, got $${bizT1.finalPrice}`
  );

  const bizT2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 20
  });
  assert(
    "Business Pick 1 service @ Tier 2 (25% off) = $150 USD",
    bizT2.finalPrice === 150 && bizT2.totalSavings === 50,
    `Expected $150 with $50 savings, got $${bizT2.finalPrice}`
  );

  const bizT3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 20
  });
  assert(
    "Business Pick 1 service @ Tier 3 (50% off) = $100 USD",
    bizT3.finalPrice === 100 && bizT3.totalSavings === 100,
    `Expected $100 with $100 savings, got $${bizT3.finalPrice}`
  );

  const bizT5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 20
  });
  assert(
    "Business Pick 1 service @ Tier 5 (Academia 75% off) = $50 USD",
    bizT5.finalPrice === 50 && bizT5.totalSavings === 150,
    `Expected $50 with $150 savings, got $${bizT5.finalPrice}`
  );

  const biz5PicksT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 5).map(s => s.slug),
    participants: 20
  });
  assert(
    "Business Pick 5 services @ Tier 1 = $1,000 USD (flagshipCheaper = false)",
    biz5PicksT1.finalPrice === 1000 && biz5PicksT1.flagshipCheaper === false,
    `Expected $1,000 and flagshipCheaper=false, got $${biz5PicksT1.finalPrice}`
  );

  const biz9Picks = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 9).map(s => s.slug),
    participants: 20
  });
  assert(
    "Business Pick 9 services @ Tier 1 = $1,800 USD (flagshipCheaper = false)",
    biz9Picks.finalPrice === 1800 && biz9Picks.flagshipCheaper === false,
    `Expected $1,800 and flagshipCheaper=false, got $${biz9Picks.finalPrice}`
  );

  const biz10Picks = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 10).map(s => s.slug),
    participants: 20
  });
  assert(
    "Business Pick 10 services @ Tier 1 = $2,000 USD (triggers flagship nudge)",
    biz10Picks.finalPrice === 2000 && biz10Picks.flagshipCheaper === true && biz10Picks.flagshipPrice === 2000,
    `Expected $2,000 and flagshipCheaper=true, got price=${biz10Picks.finalPrice}, cheaper=${biz10Picks.flagshipCheaper}`
  );

  const bizAll12 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.map(s => s.slug),
    participants: 20
  });
  assert(
    "Business Pick all 12 services @ Tier 1 = $2,400 USD (triggers flagship nudge)",
    bizAll12.finalPrice === 2400 && bizAll12.serviceCount === 12 && bizAll12.flagshipCheaper === true,
    `Expected $2,400, got $${bizAll12.finalPrice}`
  );

  const bizAll12T3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.map(s => s.slug),
    participants: 20
  });
  assert(
    "Business Pick all 12 services @ Tier 3 = $1,200 USD ($2,400 base * 0.50)",
    bizAll12T3.finalPrice === 1200 && bizAll12T3.totalSavings === 1200,
    `Expected $1,200, got $${bizAll12T3.finalPrice}`
  );

  // ---------------------------------------------------------------------------
  // SECTION 5: Business Extra Participant Scaling Tests (+ $100/seat scaled by tier)
  // ---------------------------------------------------------------------------
  const bizT1Extra2 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 22
  });
  assert(
    "Business 22 seats (2 extra @ $100) @ Tier 1 = $400 USD ($200 base + $200 extra)",
    bizT1Extra2.finalPrice === 400 && bizT1Extra2.extraParticipants === 2 && bizT1Extra2.extraHeadRate === 200,
    `Expected $400 with $200 extra, got $${bizT1Extra2.finalPrice}`
  );

  const bizT2Extra2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 22 // 2 extra heads * $100 * 0.75 = $150; base = $200 * 0.75 = $150; total = $300
  });
  assert(
    "Business 22 seats @ Tier 2 (25% parity scaled) = $300 USD ($150 base + $150 extra)",
    bizT2Extra2.finalPrice === 300 && bizT2Extra2.extraHeadRate === 150,
    `Expected $300 with $150 extra, got $${bizT2Extra2.finalPrice}`
  );

  const bizT3Extra5 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 25 // 5 extra heads * $100 * 0.50 = $250; base = $200 * 0.50 = $100; total = $350
  });
  assert(
    "Business 25 seats @ Tier 3 = $350 USD ($100 base + $250 extra)",
    bizT3Extra5.finalPrice === 350 && bizT3Extra5.extraHeadRate === 250,
    `Expected $350 with $250 extra, got $${bizT3Extra5.finalPrice}`
  );

  const bizT5Extra10 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 30 // 10 extra heads * $100 * 0.25 = $250; base = $200 * 0.25 = $50; total = $300
  });
  assert(
    "Business 30 seats @ Tier 5 = $300 USD ($50 base + $250 extra)",
    bizT5Extra10.finalPrice === 300 && bizT5Extra10.extraHeadRate === 250,
    `Expected $300 with $250 extra, got $${bizT5Extra10.finalPrice}`
  );

  const biz50Seats = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 50 // 30 extra heads * $100 = $3,000; base = $2,000; total = $5,000
  });
  assert(
    "Business Flagship 50 seats @ Tier 1 = $5,000 USD ($2,000 base + $3,000 extra)",
    biz50Seats.finalPrice === 5000 && biz50Seats.extraParticipants === 30 && biz50Seats.extraHeadRate === 3000,
    `Expected $5,000 with 30 extra, got $${biz50Seats.finalPrice}`
  );

  // ---------------------------------------------------------------------------
  // SECTION 6: Flagship Package Tests (Business $2,000 / Personal $1,000 Base)
  // ---------------------------------------------------------------------------
  const flagshipBizT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 20
  });
  assert(
    "Business Flagship @ Tier 1 = $2,000 USD (0% off)",
    flagshipBizT1.finalPrice === 2000 && flagshipBizT1.basePrice === 2000 && flagshipBizT1.totalSavings === 0,
    `Expected $2,000, got $${flagshipBizT1.finalPrice}`
  );

  const flagshipBizT2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 20
  });
  assert(
    "Business Flagship @ Tier 2 = $1,500 USD (25% off, $500 savings)",
    flagshipBizT2.finalPrice === 1500 && flagshipBizT2.totalSavings === 500,
    `Expected $1,500 with $500 savings, got $${flagshipBizT2.finalPrice}`
  );

  const flagshipBizT3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 20
  });
  assert(
    "Business Flagship @ Tier 3 = $1,000 USD (50% off, $1,000 savings)",
    flagshipBizT3.finalPrice === 1000 && flagshipBizT3.totalSavings === 1000,
    `Expected $1,000 with $1,000 savings, got $${flagshipBizT3.finalPrice}`
  );

  const flagshipBizT5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 20
  });
  assert(
    "Business Flagship @ Tier 5 = $500 USD (75% off, $1,500 savings)",
    flagshipBizT5.finalPrice === 500 && flagshipBizT5.totalSavings === 1500,
    `Expected $500 with $1,500 savings, got $${flagshipBizT5.finalPrice}`
  );

  const flagshipPersonalT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pkg-personal",
    participants: 1
  });
  assert(
    "Personal Flagship @ Tier 1 = $1,000 USD (0% off)",
    flagshipPersonalT1.finalPrice === 1000 && flagshipPersonalT1.basePrice === 1000 && flagshipPersonalT1.totalSavings === 0,
    `Expected $1,000, got $${flagshipPersonalT1.finalPrice}`
  );

  const flagshipPersonalT2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "individual",
    packageType: "pkg-personal",
    participants: 1
  });
  assert(
    "Personal Flagship @ Tier 2 = $750 USD (25% off, $250 savings)",
    flagshipPersonalT2.finalPrice === 750 && flagshipPersonalT2.totalSavings === 250,
    `Expected $750 with $250 savings, got $${flagshipPersonalT2.finalPrice}`
  );

  const flagshipPersonalT3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "individual",
    packageType: "pkg-personal",
    participants: 1
  });
  assert(
    "Personal Flagship @ Tier 3 = $500 USD (50% off, $500 savings)",
    flagshipPersonalT3.finalPrice === 500 && flagshipPersonalT3.totalSavings === 500,
    `Expected $500 with $500 savings, got $${flagshipPersonalT3.finalPrice}`
  );

  const flagshipPersonalT5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "individual",
    packageType: "pkg-personal",
    participants: 1
  });
  assert(
    "Personal Flagship @ Tier 5 = $250 USD (75% off, $750 savings)",
    flagshipPersonalT5.finalPrice === 250 && flagshipPersonalT5.totalSavings === 750,
    `Expected $250 with $750 savings, got $${flagshipPersonalT5.finalPrice}`
  );

  // ---------------------------------------------------------------------------
  // SECTION 7: Custom Amount Package Tests
  // ---------------------------------------------------------------------------
  const custom5000T1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "custom",
    customBase: 5000,
    participants: 20
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
    participants: 20
  });
  assert(
    "Custom scope $5,000 @ Tier 2 (25% off) = $3,750 USD",
    custom5000T2.finalPrice === 3750 && custom5000T2.totalSavings === 1250,
    `Expected $3,750, got $${custom5000T2.finalPrice}`
  );

  const custom5000T5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "business",
    packageType: "custom",
    customBase: 5000,
    participants: 20
  });
  assert(
    "Custom scope $5,000 @ Tier 5 (75% off) = $1,250 USD",
    custom5000T5.finalPrice === 1250 && custom5000T5.totalSavings === 3750,
    `Expected $1,250, got $${custom5000T5.finalPrice}`
  );

  const customZero = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "custom",
    customBase: 0,
    participants: 20
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
    participants: 20
  });
  assert(
    "Custom scope negative amount is clamped to $0 USD",
    customNegative.finalPrice === 0 && customNegative.basePrice === 0,
    `Expected $0, got $${customNegative.finalPrice}`
  );

  // ---------------------------------------------------------------------------
  // SECTION 8: Defensive Edge Cases & Malformed Inputs Tests
  // ---------------------------------------------------------------------------
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
    "NaN or negative participants defaults to 20 for business and 1 for individual",
    PricingCalculator.calculatePricing({ clientMode: "business", participants: "invalid" }).participants === 20 &&
    PricingCalculator.calculatePricing({ clientMode: "individual", participants: "invalid" }).participants === 1,
    "Malformed participants were not safely normalized"
  );

  // ---------------------------------------------------------------------------
  // SECTION 9: DOM Controller Simulated Live-Update Tests (pricing.html Pane)
  // Verifies that the price listed changes as soon as a change is made in ANY UI element
  // ---------------------------------------------------------------------------
  console.log("-------------------------------------------------");
  console.log("TESTING DOM CONTROLLER & REAL-TIME UI PANE CHANGES");
  console.log("-------------------------------------------------");

  function setupMockDOM() {
    const elements = {};

    function makeEl(tag, id = "", classes = []) {
      const classListSet = new Set(classes);
      const listeners = {};
      const attributes = {};

      const el = {
        tagName: tag.toUpperCase(),
        id: id,
        classList: {
          add: (...cls) => cls.forEach(c => classListSet.add(c)),
          remove: (...cls) => cls.forEach(c => classListSet.delete(c)),
          contains: (c) => classListSet.has(c)
        },
        style: {},
        attributes: attributes,
        getAttribute: (name) => attributes[name] || null,
        setAttribute: (name, val) => { attributes[name] = String(val); },
        addEventListener: (evt, handler) => {
          if (!listeners[evt]) listeners[evt] = [];
          listeners[evt].push(handler);
        },
        dispatchEvent: (evt) => {
          const type = typeof evt === 'string' ? evt : evt.type;
          if (listeners[type]) {
            listeners[type].forEach(fn => fn.call(el, evt));
          }
        },
        trigger: (evtType, customProps = {}) => {
          const evt = Object.assign({ type: evtType, target: el }, customProps);
          if (listeners[evtType]) {
            listeners[evtType].forEach(fn => fn.call(el, evt));
          }
        },
        value: "",
        textContent: "",
        innerHTML: "",
        checked: false,
        disabled: false,
        focus: () => {},
        select: () => {},
        scrollIntoView: () => {},
        querySelector: () => null,
        querySelectorAll: () => []
      };

      if (id) elements[id] = el;
      return el;
    }

    const btnModeBusiness = makeEl("button", "btn-mode-business", ["calc-tier-btn", "active"]);
    const btnModeIndividual = makeEl("button", "btn-mode-individual", ["calc-tier-btn"]);
    const calcPackage = makeEl("select", "calc-package");
    const optPkgBiz = makeEl("option"); optPkgBiz.value = "pkg-business"; optPkgBiz.setAttribute("data-price", "2000");
    const optPkgPers = makeEl("option"); optPkgPers.value = "pkg-personal"; optPkgPers.setAttribute("data-price", "1000");
    const optPick = makeEl("option"); optPick.value = "pick-services"; optPick.setAttribute("data-price", "200");
    const optCustom = makeEl("option"); optCustom.value = "custom"; optCustom.setAttribute("data-price", "");
    calcPackage.options = [optPkgBiz, optPkgPers, optPick, optCustom];
    calcPackage.selectedIndex = 0;
    calcPackage.value = "pkg-business";

    const servicesMultiSelectWrap = makeEl("div", "services-multi-select-wrap");
    const servicesCheckboxGrid = makeEl("div", "services-checkbox-grid");
    const selectedServiceCountLabel = makeEl("span", "selected-service-count-label");
    const btnSelectAllServices = makeEl("button", "btn-select-all-services");
    const btnClearServices = makeEl("button", "btn-clear-services");
    const calcCustomAmount = makeEl("input", "calc-custom-amount"); calcCustomAmount.value = "2000";
    const calcParticipants = makeEl("input", "calc-participants"); calcParticipants.value = "20";
    const calcParticipantsHint = makeEl("span", "calc-participants-hint");
    const calcParticipantsDesc = makeEl("p", "calc-participants-desc");
    const calcScopeVal = makeEl("span", "calc-scope-val");
    const calcCohortVal = makeEl("span", "calc-cohort-val");

    const calcCountry = makeEl("select", "calc-country");
    const optUS = makeEl("option"); optUS.value = "US"; optUS.setAttribute("data-tier", "1");
    const optBR = makeEl("option"); optBR.value = "BR"; optBR.setAttribute("data-tier", "2");
    const optIN = makeEl("option"); optIN.value = "IN"; optIN.setAttribute("data-tier", "3");
    const optSTART = makeEl("option"); optSTART.value = "STARTUP_PRE"; optSTART.setAttribute("data-tier", "4");
    const optUNI = makeEl("option"); optUNI.value = "UNI_ACAD"; optUNI.setAttribute("data-tier", "5");
    const optLDC = makeEl("option"); optLDC.value = "LDC_CRISIS"; optLDC.setAttribute("data-tier", "6");
    calcCountry.options = [optUS, optBR, optIN, optSTART, optUNI, optLDC];
    calcCountry.selectedIndex = 0;
    calcCountry.value = "US";
    calcCountry.querySelector = (sel) => {
      const match = sel.match(/data-tier="(\d+)"/);
      if (match) {
        return calcCountry.options.find(o => o.getAttribute("data-tier") === match[1]) || null;
      }
      return null;
    };
    calcCountry.selectedOptions = [optUS];

    const tierBtns = [1, 2, 3, 4, 5, 6].map(t => {
      const b = makeEl("button", "", ["calc-tier-btn", t === 1 ? "active" : ""]);
      b.setAttribute("data-tier", String(t));
      return b;
    });

    const calcResultBox = makeEl("div", "calc-result-box");
    const calcTierBadge = makeEl("span", "calc-tier-badge");
    const calcDisplayPrice = makeEl("div", "calc-display-price");
    const calcDisplayNote = makeEl("div", "calc-display-note");
    const calcSavingsBadge = makeEl("div", "calc-savings-badge");
    const calcFlagshipNudge = makeEl("div", "calc-flagship-nudge");
    const calcBaseValue = makeEl("span", "calc-base-value");
    const calcAppliedTier = makeEl("span", "calc-applied-tier");
    const calcAdjustmentVal = makeEl("span", "calc-adjustment-val");
    const calcCtaBtn = makeEl("a", "calc-cta-btn");

    const serviceCards = PricingCalculator.ALL_SERVICES_DATA.map(s => {
      const card = makeEl("div", "", ["price-card-faq"]);
      const indivPrice = makeEl("span", "", ["svc-indiv-price"]);
      const bizPrice = makeEl("span", "", ["svc-biz-price"]);
      const indivTiers = makeEl("span", "", ["svc-indiv-tiers"]);
      const bizTiers = makeEl("span", "", ["svc-biz-tiers"]);
      card.querySelector = (sel) => {
        if (sel === ".svc-indiv-price") return indivPrice;
        if (sel === ".svc-biz-price") return bizPrice;
        if (sel === ".svc-indiv-tiers") return indivTiers;
        if (sel === ".svc-biz-tiers") return bizTiers;
        return null;
      };
      return card;
    });

    let renderedCheckboxes = [];
    Object.defineProperty(servicesCheckboxGrid, "innerHTML", {
      get: () => "",
      set: (html) => {
        renderedCheckboxes = PricingCalculator.ALL_SERVICES_DATA.map(s => {
          const cb = makeEl("input", "", ["service-pick-cb"]);
          cb.value = s.slug;
          cb.checked = (s.slug === "service-generative-ai-transformation");
          return cb;
        });
      }
    });

    servicesCheckboxGrid.querySelectorAll = (sel) => {
      if (sel === ".service-pick-cb") return renderedCheckboxes;
      return [];
    };

    const mockDocument = {
      readyState: "complete",
      getElementById: (id) => elements[id] || null,
      querySelectorAll: (sel) => {
        if (sel === ".calc-tier-btn") return tierBtns;
        if (sel.includes("#individual-services-list")) return serviceCards;
        return [];
      },
      addEventListener: () => {}
    };

    return { mockDocument, elements, tierBtns, serviceCards, getCheckboxes: () => renderedCheckboxes };
  }

  const { mockDocument, elements, tierBtns, serviceCards, getCheckboxes } = setupMockDOM();
  const originalDoc = typeof document !== 'undefined' ? document : undefined;
  const originalWin = typeof window !== 'undefined' ? window : undefined;

  global.document = mockDocument;
  global.window = { document: mockDocument };

  // Initialize DOM Controller
  PricingCalculator.initDOMController();

  assert(
    "DOM UI Initial State: Defaults to Tier 1 Business Flagship $2,000 USD",
    elements["calc-display-price"].textContent === "$2,000 USD" && elements["calc-base-value"].textContent === "$2,000 USD",
    `Expected $2,000 USD, got ${elements["calc-display-price"].textContent}`
  );

  // 1. Client Mode Button Change: Click Individual
  elements["btn-mode-individual"].trigger("click");
  assert(
    "UI Mode Switch: Clicking Individual immediately updates price to $1,000 USD and disables participants",
    elements["calc-display-price"].textContent === "$1,000 USD" && elements["calc-participants"].disabled === true,
    `Expected $1,000 USD and disabled=true, got ${elements["calc-display-price"].textContent}, disabled=${elements["calc-participants"].disabled}`
  );

  // 2. Client Mode Button Change: Click Business
  elements["btn-mode-business"].trigger("click");
  assert(
    "UI Mode Switch: Clicking Business immediately updates price to $2,000 USD and enables participants",
    elements["calc-display-price"].textContent === "$2,000 USD" && elements["calc-participants"].disabled === false,
    `Expected $2,000 USD and disabled=false, got ${elements["calc-display-price"].textContent}, disabled=${elements["calc-participants"].disabled}`
  );

  // 3. Package Select Change: Switch to pick-services
  elements["calc-package"].value = "pick-services";
  elements["calc-package"].selectedIndex = 2;
  elements["calc-package"].trigger("change");
  assert(
    "UI Package Change: Switching to pick-services immediately calculates 1 selected service ($200 USD)",
    elements["calc-display-price"].textContent === "$200 USD" && elements["services-multi-select-wrap"].style.display === "block",
    `Expected $200 USD and visible multi-select wrap, got ${elements["calc-display-price"].textContent}`
  );

  // 4. Clear Button Click: Nothing Selected Case
  elements["btn-clear-services"].trigger("click");
  assert(
    "UI Clear Services: Clicking Clear immediately unchecks all services and displays $0 USD",
    elements["calc-display-price"].textContent === "$0 USD" &&
    elements["calc-base-value"].textContent === "$0 USD" &&
    elements["selected-service-count-label"].textContent.includes("0 Services Selected ($0 USD)"),
    `Expected $0 USD, got price=${elements["calc-display-price"].textContent}, base=${elements["calc-base-value"].textContent}, label=${elements["selected-service-count-label"].textContent}`
  );

  // 5. Checkbox Interaction: Check 1 service
  const checkboxes = getCheckboxes();
  if (checkboxes.length > 0) {
    checkboxes[0].checked = true;
    checkboxes[0].trigger("change");
    assert(
      "UI Checkbox Check: Toggling a service immediately updates price from $0 USD to $200 USD",
      elements["calc-display-price"].textContent === "$200 USD",
      `Expected $200 USD, got ${elements["calc-display-price"].textContent}`
    );

    // Check a second service
    if (checkboxes.length > 1) {
      checkboxes[1].checked = true;
      checkboxes[1].trigger("change");
      assert(
        "UI Checkbox Check: Toggling second service immediately updates price to $400 USD",
        elements["calc-display-price"].textContent === "$400 USD",
        `Expected $400 USD, got ${elements["calc-display-price"].textContent}`
      );
    }
  }

  // 6. Select All Services Button Click
  elements["btn-select-all-services"].trigger("click");
  assert(
    "UI Select All: Clicking All 12 immediately checks all services and updates price to $2,400 USD",
    elements["calc-display-price"].textContent === "$2,400 USD" &&
    elements["selected-service-count-label"].textContent.includes("12 Services Selected"),
    `Expected $2,400 USD, got ${elements["calc-display-price"].textContent}`
  );

  // 7. Participants Input Change: Extra head scaling (+5 participants)
  elements["calc-participants"].value = "25";
  elements["calc-participants"].trigger("input");
  assert(
    "UI Participants Input: Changing participants to 25 immediately adds $500 extra head fee ($2,900 USD)",
    elements["calc-display-price"].textContent === "$2,900 USD",
    `Expected $2,900 USD, got ${elements["calc-display-price"].textContent}`
  );

  // 8. Custom Base Amount Input Change: Typing 5000
  elements["calc-custom-amount"].value = "5000";
  elements["calc-custom-amount"].trigger("input");
  assert(
    "UI Custom Amount Input: Typing $5,000 immediately recalculates price ($5,500 USD with 5 extra seats)",
    elements["calc-display-price"].textContent === "$5,500 USD",
    `Expected $5,500 USD, got ${elements["calc-display-price"].textContent}`
  );

  // 9. Custom Base Amount Input Change: Typing 0
  elements["calc-custom-amount"].value = "0";
  elements["calc-custom-amount"].trigger("input");
  assert(
    "UI Custom Amount Input: Typing $0 immediately displays $0 USD",
    elements["calc-display-price"].textContent === "$0 USD",
    `Expected $0 USD, got ${elements["calc-display-price"].textContent}`
  );

  // 10. Country Dropdown Change: Selecting India (Tier 3, 50% discount)
  elements["calc-custom-amount"].value = "2000";
  elements["calc-custom-amount"].trigger("input");
  elements["calc-country"].selectedIndex = 2; // IN (data-tier="3")
  elements["calc-country"].selectedOptions = [elements["calc-country"].options[2]];
  elements["calc-country"].trigger("change");
  assert(
    "UI Country Dropdown: Selecting Tier 3 country (India) immediately updates parity tier and price",
    elements["calc-applied-tier"].textContent.includes("Tier 3") &&
    elements["calc-display-price"].textContent === "$1,250 USD", // ($2,000 base + $500 extra) * 0.50 = $1,250
    `Expected Tier 3 and $1,250 USD, got tier=${elements["calc-applied-tier"].textContent}, price=${elements["calc-display-price"].textContent}`
  );

  // 11. Tier Pill Button Click: Clicking Tier 5 (75% discount)
  tierBtns[4].trigger("click"); // Tier 5 button
  assert(
    "UI Tier Button Click: Clicking Tier 5 button immediately updates active tier and applies 75% discount",
    elements["calc-applied-tier"].textContent.includes("Tier 5") &&
    elements["calc-display-price"].textContent === "$625 USD", // ($2,000 base + $500 extra) * 0.25 = $625
    `Expected Tier 5 and $625 USD, got tier=${elements["calc-applied-tier"].textContent}, price=${elements["calc-display-price"].textContent}`
  );

  // 12. Tier Pill Button Click: Clicking Tier 6 (Subsidized custom quote)
  tierBtns[5].trigger("click"); // Tier 6 button
  assert(
    "UI Tier Button Click: Clicking Tier 6 button immediately displays custom subsidized quote message",
    elements["calc-display-price"].textContent.includes("Custom subsidized quote"),
    `Expected custom quote message, got ${elements["calc-display-price"].textContent}`
  );

  // 13. Dynamic update of individual service cards across the page
  assert(
    "Service Cards Parity Update: Service cards reflect active tier pricing dynamically",
    serviceCards[0].querySelector(".svc-indiv-price").textContent.includes("Custom quote"),
    `Expected custom quote on service card for Tier 6, got ${serviceCards[0].querySelector(".svc-indiv-price").textContent}`
  );

  // Switch back to Tier 2 to verify dynamic price rendering on cards
  tierBtns[1].trigger("click");
  assert(
    "Service Cards Parity Update: Switching to Tier 2 renders calibrated prices on service cards",
    serviceCards[0].querySelector(".svc-indiv-price").textContent.includes("$75 USD (T2)") &&
    serviceCards[0].querySelector(".svc-biz-price").textContent.includes("$150 USD (T2)"),
    `Expected $75/$150 for Tier 2 on service cards, got indiv=${serviceCards[0].querySelector(".svc-indiv-price").textContent}, biz=${serviceCards[0].querySelector(".svc-biz-price").textContent}`
  );

  // Restore globals
  if (originalDoc !== undefined) global.document = originalDoc; else delete global.document;
  if (originalWin !== undefined) global.window = originalWin; else delete global.window;

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
