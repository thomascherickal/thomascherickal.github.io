/**
 * pricing-tests.js
 * Comprehensive Unit and Functional Test Suite for Pricing Calculator & Dynamic UI Pane
 * Runs under Node.js or browser console with ZERO external dependencies.
 */

const PricingCalculator = typeof require !== 'undefined'
  ? require('./pricing-calculator.js')
  : window.PricingCalculator;

// --- AUTHORITATIVE SYNCHRONIZED PRICING FACTS ---
const B_FLAG_PRICE = 3000;
const P_FLAG_PRICE = 3000;
const B_IND_PRICE = 300;
const B_EXTRA_SEAT = 300;
const B_INCLUDED_SEATS = 10;
// --- END AUTHORITATIVE SYNCHRONIZED PRICING FACTS ---

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
  // SECTION 3: Individual Pick-Services Tests (strictly 1 person)
  // ---------------------------------------------------------------------------
  const indivT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    `Individual Pick 1 service @ Tier 1 = $${B_IND_PRICE} USD (0% off)`,
    indivT1.finalPrice === B_IND_PRICE && indivT1.participants === 1 && indivT1.totalSavings === 0,
    `Expected $${B_IND_PRICE} with 1 participant, got $${indivT1.finalPrice}`
  );

  const indivT2Expected = Math.round(B_IND_PRICE * 0.75);
  const indivT2Savings = B_IND_PRICE - indivT2Expected;
  const indivT2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    `Individual Pick 1 service @ Tier 2 (25% off) = $${indivT2Expected} USD`,
    indivT2.finalPrice === indivT2Expected && indivT2.totalSavings === indivT2Savings,
    `Expected $${indivT2Expected} price and $${indivT2Savings} savings, got price $${indivT2.finalPrice}`
  );

  const indivT3Expected = Math.round(B_IND_PRICE * 0.50);
  const indivT3Savings = B_IND_PRICE - indivT3Expected;
  const indivT3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    `Individual Pick 1 service @ Tier 3 (50% off) = $${indivT3Expected} USD`,
    indivT3.finalPrice === indivT3Expected && indivT3.totalSavings === indivT3Savings,
    `Expected $${indivT3Expected} price and $${indivT3Savings} savings, got price $${indivT3.finalPrice}`
  );

  const indivT4Expected = Math.round(B_IND_PRICE * 0.50);
  const indivT4Savings = B_IND_PRICE - indivT4Expected;
  const indivT4 = PricingCalculator.calculatePricing({
    tierId: 4,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    `Individual Pick 1 service @ Tier 4 (Startups 50% off) = $${indivT4Expected} USD`,
    indivT4.finalPrice === indivT4Expected && indivT4.totalSavings === indivT4Savings,
    `Expected $${indivT4Expected} price and $${indivT4Savings} savings, got price $${indivT4.finalPrice}`
  );

  const indivT5Expected = Math.round(B_IND_PRICE * 0.25);
  const indivT5Savings = B_IND_PRICE - indivT5Expected;
  const indivT5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 1
  });
  assert(
    `Individual Pick 1 service @ Tier 5 (Academia 75% off) = $${indivT5Expected} USD`,
    indivT5.finalPrice === indivT5Expected && indivT5.totalSavings === indivT5Savings,
    `Expected $${indivT5Expected} price and $${indivT5Savings} savings, got price $${indivT5.finalPrice}`
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
    `Individual Pick 3 services @ Tier 1 = $${3 * B_IND_PRICE} USD`,
    indivMulti3.finalPrice === 3 * B_IND_PRICE && indivMulti3.serviceCount === 3,
    `Expected $${3 * B_IND_PRICE} for 3 services, got $${indivMulti3.finalPrice}`
  );

  const indivMulti5T2Base = 5 * B_IND_PRICE;
  const indivMulti5T2Expected = Math.round(indivMulti5T2Base * 0.75);
  const indivMulti5T2Savings = indivMulti5T2Base - indivMulti5T2Expected;
  const indivMulti5T2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 5).map(s => s.slug),
    participants: 1
  });
  assert(
    `Individual Pick 5 services @ Tier 2 = $${indivMulti5T2Expected} USD ($${indivMulti5T2Base.toLocaleString()} base * 0.75)`,
    indivMulti5T2.finalPrice === indivMulti5T2Expected && indivMulti5T2.totalSavings === indivMulti5T2Savings,
    `Expected $${indivMulti5T2Expected} with $${indivMulti5T2Savings} savings, got $${indivMulti5T2.finalPrice}`
  );

  const indivMulti9 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 9).map(s => s.slug),
    participants: 1
  });
  assert(
    `Individual Pick 9 services ($${(9 * B_IND_PRICE).toLocaleString()} USD) does not trigger flagship nudge`,
    indivMulti9.finalPrice === 9 * B_IND_PRICE && indivMulti9.flagshipCheaper === false,
    `Expected $${9 * B_IND_PRICE} and flagshipCheaper=false, got price=${indivMulti9.finalPrice}, cheaper=${indivMulti9.flagshipCheaper}`
  );

  const indivMulti10 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 10).map(s => s.slug),
    participants: 1
  });
  assert(
    `Individual Pick 10 services ($${(10 * B_IND_PRICE).toLocaleString()} USD) triggers flagship nudge ($${B_FLAG_PRICE.toLocaleString()} threshold)`,
    indivMulti10.finalPrice === 10 * B_IND_PRICE && indivMulti10.flagshipCheaper === true && indivMulti10.flagshipPrice === B_FLAG_PRICE,
    `Expected $${10 * B_IND_PRICE} and flagshipCheaper=true, got price=${indivMulti10.finalPrice}, cheaper=${indivMulti10.flagshipCheaper}`
  );

  const indivAll12 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.map(s => s.slug),
    participants: 1
  });
  assert(
    `Individual Pick all 12 services @ Tier 1 = $${(12 * B_IND_PRICE).toLocaleString()} USD (triggers flagship nudge)`,
    indivAll12.finalPrice === 12 * B_IND_PRICE && indivAll12.serviceCount === 12 && indivAll12.flagshipCheaper === true,
    `Expected $${12 * B_IND_PRICE}, got $${indivAll12.finalPrice}`
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
  // SECTION 4: Business Pick-Services Tests (up to 10 seats)
  // ---------------------------------------------------------------------------
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
  const bizT2Savings = B_IND_PRICE - bizT2Expected;
  const bizT2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 10
  });
  assert(
    `Business Pick 1 service @ Tier 2 (25% off) = $${bizT2Expected} USD`,
    bizT2.finalPrice === bizT2Expected && bizT2.totalSavings === bizT2Savings,
    `Expected $${bizT2Expected} with $${bizT2Savings} savings, got $${bizT2.finalPrice}`
  );

  const bizT3Expected = Math.round(B_IND_PRICE * 0.50);
  const bizT3Savings = B_IND_PRICE - bizT3Expected;
  const bizT3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 10
  });
  assert(
    `Business Pick 1 service @ Tier 3 (50% off) = $${bizT3Expected} USD`,
    bizT3.finalPrice === bizT3Expected && bizT3.totalSavings === bizT3Savings,
    `Expected $${bizT3Expected} with $${bizT3Savings} savings, got $${bizT3.finalPrice}`
  );

  const bizT5Expected = Math.round(B_IND_PRICE * 0.25);
  const bizT5Savings = B_IND_PRICE - bizT5Expected;
  const bizT5 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 10
  });
  assert(
    `Business Pick 1 service @ Tier 5 (Academia 75% off) = $${bizT5Expected} USD`,
    bizT5.finalPrice === bizT5Expected && bizT5.totalSavings === bizT5Savings,
    `Expected $${bizT5Expected} with $${bizT5Savings} savings, got $${bizT5.finalPrice}`
  );

  const biz5PicksT1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 5).map(s => s.slug),
    participants: 10
  });
  assert(
    `Business Pick 5 services @ Tier 1 = $${(5 * B_IND_PRICE).toLocaleString()} USD (flagshipCheaper = false)`,
    biz5PicksT1.finalPrice === 5 * B_IND_PRICE && biz5PicksT1.flagshipCheaper === false,
    `Expected $${5 * B_IND_PRICE} and flagshipCheaper=false, got $${biz5PicksT1.finalPrice}`
  );

  const biz9Picks = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 9).map(s => s.slug),
    participants: 10
  });
  assert(
    `Business Pick 9 services @ Tier 1 = $${(9 * B_IND_PRICE).toLocaleString()} USD (flagshipCheaper = false)`,
    biz9Picks.finalPrice === 9 * B_IND_PRICE && biz9Picks.flagshipCheaper === false,
    `Expected $${9 * B_IND_PRICE} and flagshipCheaper=false, got $${biz9Picks.finalPrice}`
  );

  const biz10Picks = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.slice(0, 10).map(s => s.slug),
    participants: 10
  });
  assert(
    `Business Pick 10 services @ Tier 1 = $${(10 * B_IND_PRICE).toLocaleString()} USD (triggers flagship nudge)`,
    biz10Picks.finalPrice === 10 * B_IND_PRICE && biz10Picks.flagshipCheaper === true && biz10Picks.flagshipPrice === B_FLAG_PRICE,
    `Expected $${10 * B_IND_PRICE} and flagshipCheaper=true, got price=${biz10Picks.finalPrice}, cheaper=${biz10Picks.flagshipCheaper}`
  );

  const bizAll12 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.map(s => s.slug),
    participants: 10
  });
  assert(
    `Business Pick all 12 services @ Tier 1 = $${(12 * B_IND_PRICE).toLocaleString()} USD (triggers flagship nudge)`,
    bizAll12.finalPrice === 12 * B_IND_PRICE && bizAll12.serviceCount === 12 && bizAll12.flagshipCheaper === true,
    `Expected $${12 * B_IND_PRICE}, got $${bizAll12.finalPrice}`
  );

  const bizAll12T3Base = 12 * B_IND_PRICE;
  const bizAll12T3Expected = Math.round(bizAll12T3Base * 0.50);
  const bizAll12T3Savings = bizAll12T3Base - bizAll12T3Expected;
  const bizAll12T3 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: PricingCalculator.ALL_SERVICES_DATA.map(s => s.slug),
    participants: 10
  });
  assert(
    `Business Pick all 12 services @ Tier 3 = $${bizAll12T3Expected.toLocaleString()} USD ($${bizAll12T3Base.toLocaleString()} base * 0.50)`,
    bizAll12T3.finalPrice === bizAll12T3Expected && bizAll12T3.totalSavings === bizAll12T3Savings,
    `Expected $${bizAll12T3Expected}, got $${bizAll12T3.finalPrice}`
  );

  // ---------------------------------------------------------------------------
  // SECTION 5: Business Extra Participant Scaling Tests (+ $250/seat scaled by tier)
  // ---------------------------------------------------------------------------
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
    bizT1Extra2.finalPrice === bizT1Extra2Expected && bizT1Extra2.extraParticipants === 2 && bizT1Extra2.extraHeadRate === bizT1Extra2Rate,
    `Expected $${bizT1Extra2Expected} with $${bizT1Extra2Rate} extra, got $${bizT1Extra2.finalPrice}`
  );

  const bizT2Extra2Rate = Math.round(2 * B_EXTRA_SEAT * 0.75);
  const bizT2Extra2Base = Math.round(B_IND_PRICE * 0.75);
  const bizT2Extra2Expected = bizT2Extra2Base + bizT2Extra2Rate;
  const bizT2Extra2 = PricingCalculator.calculatePricing({
    tierId: 2,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 12
  });
  assert(
    `Business 12 seats @ Tier 2 (25% parity scaled) = $${bizT2Extra2Expected} USD ($${bizT2Extra2Base} base + $${bizT2Extra2Rate} extra)`,
    bizT2Extra2.finalPrice === bizT2Extra2Expected && bizT2Extra2.extraHeadRate === bizT2Extra2Rate,
    `Expected $${bizT2Extra2Expected} with $${bizT2Extra2Rate} extra, got $${bizT2Extra2.finalPrice}`
  );

  const bizT3Extra5Rate = Math.round(5 * B_EXTRA_SEAT * 0.50);
  const bizT3Extra5Base = Math.round(B_IND_PRICE * 0.50);
  const bizT3Extra5Expected = bizT3Extra5Base + bizT3Extra5Rate;
  const bizT3Extra5 = PricingCalculator.calculatePricing({
    tierId: 3,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 15
  });
  assert(
    `Business 15 seats @ Tier 3 = $${bizT3Extra5Expected} USD ($${bizT3Extra5Base} base + $${bizT3Extra5Rate} extra)`,
    bizT3Extra5.finalPrice === bizT3Extra5Expected && bizT3Extra5.extraHeadRate === bizT3Extra5Rate,
    `Expected $${bizT3Extra5Expected} with $${bizT3Extra5Rate} extra, got $${bizT3Extra5.finalPrice}`
  );

  const bizT5Extra10Rate = Math.round(5 * B_EXTRA_SEAT * 0.25);
  const bizT5Extra10Base = Math.round(B_IND_PRICE * 0.25);
  const bizT5Extra10Expected = bizT5Extra10Base + bizT5Extra10Rate;
  const bizT5Extra10 = PricingCalculator.calculatePricing({
    tierId: 5,
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: ["service-generative-ai-transformation"],
    participants: 15
  });
  assert(
    `Business 15 seats @ Tier 5 = $${bizT5Extra10Expected} USD ($${bizT5Extra10Base} base + $${bizT5Extra10Rate} extra)`,
    bizT5Extra10.finalPrice === bizT5Extra10Expected && bizT5Extra10.extraHeadRate === bizT5Extra10Rate,
    `Expected $${bizT5Extra10Expected} with $${bizT5Extra10Rate} extra, got $${bizT5Extra10.finalPrice}`
  );

  const biz35ExtraRate = 25 * B_EXTRA_SEAT;
  const biz35Expected = B_FLAG_PRICE + biz35ExtraRate;
  const biz35Seats = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "pkg-business",
    participants: 35
  });
  assert(
    `Business Flagship 35 seats @ Tier 1 = $${biz35Expected.toLocaleString()} USD ($${B_FLAG_PRICE.toLocaleString()} base + $${biz35ExtraRate.toLocaleString()} extra)`,
    biz35Seats.finalPrice === biz35Expected && biz35Seats.extraParticipants === 25 && biz35Seats.extraHeadRate === biz35ExtraRate,
    `Expected $${biz35Expected} with 25 extra, got $${biz35Seats.finalPrice}`
  );

  // ---------------------------------------------------------------------------
  // SECTION 6: Flagship Package Tests
  // ---------------------------------------------------------------------------
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

  // ---------------------------------------------------------------------------
  // SECTION 7: Custom Amount Package Tests
  // ---------------------------------------------------------------------------
  const custom5000T1 = PricingCalculator.calculatePricing({
    tierId: 1,
    clientMode: "business",
    packageType: "custom",
    customBase: 5000,
    participants: 10
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
    participants: 10
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
    participants: 10
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
    participants: 10
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
    participants: 10
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
    "NaN or negative participants defaults to 10 for business and 1 for individual",
    PricingCalculator.calculatePricing({ clientMode: "business", participants: "invalid" }).participants === 10 &&
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
        focus: () => { },
        select: () => { },
        scrollIntoView: () => { },
        querySelector: () => null,
        querySelectorAll: () => []
      };

      if (id) elements[id] = el;
      return el;
    }

    const btnModeBusiness = makeEl("button", "btn-mode-business", ["calc-tier-btn", "active"]);
    const btnModeIndividual = makeEl("button", "btn-mode-individual", ["calc-tier-btn"]);
    const calcPackage = makeEl("select", "calc-package");
    const optPkgBiz = makeEl("option"); optPkgBiz.value = "pkg-business"; optPkgBiz.setAttribute("data-price", "2500");
    const optPkgPers = makeEl("option"); optPkgPers.value = "pkg-personal"; optPkgPers.setAttribute("data-price", "2500");
    const optPick = makeEl("option"); optPick.value = "pick-services"; optPick.setAttribute("data-price", "250");
    const optCustom = makeEl("option"); optCustom.value = "custom"; optCustom.setAttribute("data-price", "");
    calcPackage.options = [optPkgBiz, optPkgPers, optPick, optCustom];
    calcPackage.selectedIndex = 0;
    calcPackage.value = "pkg-business";

    const servicesMultiSelectWrap = makeEl("div", "services-multi-select-wrap");
    const servicesCheckboxGrid = makeEl("div", "services-checkbox-grid");
    const selectedServiceCountLabel = makeEl("span", "selected-service-count-label");
    const btnSelectAllServices = makeEl("button", "btn-select-all-services");
    const btnClearServices = makeEl("button", "btn-clear-services");
    const calcCustomAmount = makeEl("input", "calc-custom-amount"); calcCustomAmount.value = "2500";
    const calcParticipants = makeEl("input", "calc-participants"); calcParticipants.value = "10";
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

    const allCalcTierClassButtons = [btnModeBusiness, btnModeIndividual, ...tierBtns];

    const mockDocument = {
      readyState: "complete",
      getElementById: (id) => elements[id] || null,
      querySelectorAll: (sel) => {
        if (sel === ".calc-tier-btn") return allCalcTierClassButtons;
        if (sel === "#calc-tier-buttons .calc-tier-btn, .calc-tier-btn[data-tier]") return tierBtns;
        if (sel.includes("#individual-services-list")) return serviceCards;
        return [];
      },
      addEventListener: () => { }
    };

    return { mockDocument, elements, tierBtns, allCalcTierClassButtons, serviceCards, getCheckboxes: () => renderedCheckboxes };
  }

  const { mockDocument, elements, tierBtns, allCalcTierClassButtons, serviceCards, getCheckboxes } = setupMockDOM();
  const originalDoc = typeof document !== 'undefined' ? document : undefined;
  const originalWin = typeof window !== 'undefined' ? window : undefined;

  global.document = mockDocument;
  global.window = { document: mockDocument };

  // Initialize DOM Controller
  PricingCalculator.initDOMController();

  assert(
    `DOM UI Initial State: Defaults to Tier 1 Business Flagship $${B_FLAG_PRICE.toLocaleString()} USD`,
    elements["calc-display-price"].textContent === `$${B_FLAG_PRICE.toLocaleString()} USD` && elements["calc-base-value"].textContent === `$${B_FLAG_PRICE.toLocaleString()} USD`,
    `Expected $${B_FLAG_PRICE.toLocaleString()} USD, got ${elements["calc-display-price"].textContent}`
  );

  // 1. Client Mode Button Change: Click Individual
  elements["btn-mode-individual"].trigger("click");
  assert(
    `UI Mode Switch: Clicking Individual immediately updates price to $${P_FLAG_PRICE.toLocaleString()} USD and disables participants`,
    elements["calc-display-price"].textContent === `$${P_FLAG_PRICE.toLocaleString()} USD` && elements["calc-participants"].disabled === true,
    `Expected $${P_FLAG_PRICE.toLocaleString()} USD and disabled=true, got ${elements["calc-display-price"].textContent}, disabled=${elements["calc-participants"].disabled}`
  );

  // 2. Client Mode Button Change: Click Business
  elements["btn-mode-business"].trigger("click");
  assert(
    `UI Mode Switch: Clicking Business immediately updates price to $${B_FLAG_PRICE.toLocaleString()} USD and enables participants`,
    elements["calc-display-price"].textContent === `$${B_FLAG_PRICE.toLocaleString()} USD` && elements["calc-participants"].disabled === false,
    `Expected $${B_FLAG_PRICE.toLocaleString()} USD and disabled=false, got ${elements["calc-display-price"].textContent}, disabled=${elements["calc-participants"].disabled}`
  );

  // 3. Package Select Change: Switch to pick-services
  elements["calc-package"].value = "pick-services";
  elements["calc-package"].selectedIndex = 2;
  elements["calc-package"].trigger("change");
  assert(
    `UI Package Change: Switching to pick-services immediately calculates 1 selected service ($${B_IND_PRICE.toLocaleString()} USD)`,
    elements["calc-display-price"].textContent === `$${B_IND_PRICE.toLocaleString()} USD` && elements["services-multi-select-wrap"].style.display === "block",
    `Expected $${B_IND_PRICE.toLocaleString()} USD and visible multi-select wrap, got ${elements["calc-display-price"].textContent}`
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
      `UI Checkbox Check: Toggling a service immediately updates price from $0 USD to $${B_IND_PRICE.toLocaleString()} USD`,
      elements["calc-display-price"].textContent === `$${B_IND_PRICE.toLocaleString()} USD`,
      `Expected $${B_IND_PRICE.toLocaleString()} USD, got ${elements["calc-display-price"].textContent}`
    );

    // Check a second service
    if (checkboxes.length > 1) {
      checkboxes[1].checked = true;
      checkboxes[1].trigger("change");
      assert(
        `UI Checkbox Check: Toggling second service immediately updates price to $${(2 * B_IND_PRICE).toLocaleString()} USD`,
        elements["calc-display-price"].textContent === `$${(2 * B_IND_PRICE).toLocaleString()} USD`,
        `Expected $${(2 * B_IND_PRICE).toLocaleString()} USD, got ${elements["calc-display-price"].textContent}`
      );
    }
  }

  // 6. Select All Services Button Click
  elements["btn-select-all-services"].trigger("click");
  assert(
    `UI Select All: Clicking All 12 immediately checks all services and updates price to $${(12 * B_IND_PRICE).toLocaleString()} USD`,
    elements["calc-display-price"].textContent === `$${(12 * B_IND_PRICE).toLocaleString()} USD` &&
    elements["selected-service-count-label"].textContent.includes("12 Services Selected"),
    `Expected $${(12 * B_IND_PRICE).toLocaleString()} USD, got ${elements["calc-display-price"].textContent}`
  );

  // 7. Participants Input Change: Extra head scaling (+5 participants over 10 included = 15)
  elements["calc-participants"].value = "15";
  elements["calc-participants"].trigger("input");
  const ui15Price = (12 * B_IND_PRICE) + (5 * B_EXTRA_SEAT);
  assert(
    `UI Participants Input: Changing participants to 15 immediately adds $${(5 * B_EXTRA_SEAT).toLocaleString()} extra head fee ($${ui15Price.toLocaleString()} USD)`,
    elements["calc-display-price"].textContent === `$${ui15Price.toLocaleString()} USD`,
    `Expected $${ui15Price.toLocaleString()} USD, got ${elements["calc-display-price"].textContent}`
  );

  // 8. Custom Base Amount Input Change: Typing 5000
  elements["calc-custom-amount"].value = "5000";
  elements["calc-custom-amount"].trigger("input");
  const uiCustom5000Price = 5000 + (5 * B_EXTRA_SEAT);
  assert(
    `UI Custom Amount Input: Typing $5,000 immediately recalculates price ($${uiCustom5000Price.toLocaleString()} USD with 5 extra seats)`,
    elements["calc-display-price"].textContent === `$${uiCustom5000Price.toLocaleString()} USD`,
    `Expected $${uiCustom5000Price.toLocaleString()} USD, got ${elements["calc-display-price"].textContent}`
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
  elements["calc-custom-amount"].value = "2500";
  elements["calc-custom-amount"].trigger("input");
  elements["calc-country"].selectedIndex = 2; // IN (data-tier="3")
  elements["calc-country"].selectedOptions = [elements["calc-country"].options[2]];
  elements["calc-country"].trigger("change");
  const uiCountryT3Price = Math.round((B_FLAG_PRICE + 5 * B_EXTRA_SEAT) * 0.50);
  assert(
    "UI Country Dropdown: Selecting Tier 3 country (India) immediately updates parity tier and price",
    elements["calc-applied-tier"].textContent.includes("Tier 3") &&
    elements["calc-display-price"].textContent === `$${uiCountryT3Price.toLocaleString()} USD`,
    `Expected Tier 3 and $${uiCountryT3Price.toLocaleString()} USD, got tier=${elements["calc-applied-tier"].textContent}, price=${elements["calc-display-price"].textContent}`
  );

  // 11. Tier Pill Button Click: Clicking Tier 5 (75% discount)
  tierBtns[4].trigger("click"); // Tier 5 button
  const uiTier5Price = Math.round((B_FLAG_PRICE + 5 * B_EXTRA_SEAT) * 0.25);
  assert(
    "UI Tier Button Click: Clicking Tier 5 button immediately updates active tier and applies 75% discount",
    elements["calc-applied-tier"].textContent.includes("Tier 5") &&
    elements["calc-display-price"].textContent === `$${uiTier5Price.toLocaleString()} USD`,
    `Expected Tier 5 and $${uiTier5Price.toLocaleString()} USD, got tier=${elements["calc-applied-tier"].textContent}, price=${elements["calc-display-price"].textContent}`
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
  const t2CardPrice = Math.round(B_IND_PRICE * 0.75);
  assert(
    "Service Cards Parity Update: Switching to Tier 2 renders calibrated prices on service cards",
    serviceCards[0].querySelector(".svc-indiv-price").textContent.includes(`$${t2CardPrice} USD (T2)`) &&
    serviceCards[0].querySelector(".svc-biz-price").textContent.includes(`$${t2CardPrice} USD (T2)`),
    `Expected $${t2CardPrice}/$${t2CardPrice} for Tier 2 on service cards, got indiv=${serviceCards[0].querySelector(".svc-indiv-price").textContent}, biz=${serviceCards[0].querySelector(".svc-biz-price").textContent}`
  );

  // 14. Edge Case & Crash Prevention Regression Tests
  // Verifies fix for runtime exceptions at pricing-calculator.js:408 when buttons without data-tier or invalid tiers are triggered.
  let noErrorThrown = true;
  try {
    // A. Clicking mode buttons (which carry .calc-tier-btn class but no data-tier attribute)
    elements["btn-mode-business"].trigger("click");
    elements["btn-mode-individual"].trigger("click");
    elements["btn-mode-business"].trigger("click");
  } catch (err) {
    noErrorThrown = false;
  }
  assert(
    "Crash Prevention: Clicking buttons with .calc-tier-btn lacking data-tier attribute does not throw runtime TypeError",
    noErrorThrown,
    "An error was thrown when clicking mode buttons"
  );

  // B. Calling window.selectTierAndScroll with invalid/null/undefined/string/NaN arguments
  let selectTierSafe = true;
  try {
    if (typeof global.window.selectTierAndScroll === "function") {
      global.window.selectTierAndScroll(null);
      global.window.selectTierAndScroll(undefined);
      global.window.selectTierAndScroll("invalid");
      global.window.selectTierAndScroll(NaN);
      global.window.selectTierAndScroll(999);
      global.window.selectTierAndScroll(-1);
    }
  } catch (err) {
    selectTierSafe = false;
  }
  assert(
    "Crash Prevention: window.selectTierAndScroll gracefully ignores invalid or non-existent tier IDs without crashing",
    selectTierSafe,
    "selectTierAndScroll threw when passed invalid inputs"
  );

  // C. Calling calculatePricing with invalid tierId falls back safely to Tier 1
  const invalidTierFallback = PricingCalculator.calculatePricing({
    tierId: null,
    clientMode: "business",
    packageType: "pkg-business"
  });
  assert(
    `Crash Prevention: calculatePricing with null tierId safely defaults to Tier 1 pricing ($${B_FLAG_PRICE.toLocaleString()} USD)`,
    invalidTierFallback.tierId === 1 && invalidTierFallback.finalPrice === B_FLAG_PRICE,
    `Expected Tier 1 and $${B_FLAG_PRICE}, got tierId=${invalidTierFallback.tierId}, finalPrice=${invalidTierFallback.finalPrice}`
  );

  const outOfRangeTier = PricingCalculator.calculatePricing({
    tierId: 99,
    clientMode: "business",
    packageType: "pkg-business"
  });
  assert(
    `Crash Prevention: calculatePricing with tierId=99 safely defaults to Tier 1 pricing ($${B_FLAG_PRICE.toLocaleString()} USD)`,
    outOfRangeTier.tierId === 1 && outOfRangeTier.finalPrice === B_FLAG_PRICE,
    `Expected Tier 1 and $${B_FLAG_PRICE}, got tierId=${outOfRangeTier.tierId}, finalPrice=${outOfRangeTier.finalPrice}`
  );

  // D. End-to-end full sequence workflow: Mode Switch -> Pick Services -> Change Country -> Change Participants -> Click Tier Button
  let e2eWorkflowPassed = true;
  try {
    // Step 1: Switch to individual
    elements["btn-mode-individual"].trigger("click");
    // Step 2: Switch to pick services
    elements["calc-package"].value = "pick-services";
    elements["calc-package"].trigger("change");
    // Step 3: Check 3 services
    const cbs = getCheckboxes();
    cbs[0].checked = true;
    cbs[1].checked = true;
    cbs[2].checked = true;
    cbs[0].trigger("change");
    // Step 4: Country to Tier 4 (Startups 50% off)
    elements["calc-country"].selectedIndex = 3;
    elements["calc-country"].selectedOptions = [elements["calc-country"].options[3]];
    elements["calc-country"].trigger("change");
    // Step 5: Click Tier 3 pill button
    tierBtns[2].trigger("click"); // Tier 3
  } catch (err) {
    e2eWorkflowPassed = false;
  }
  assert(
    "End-to-End Workflow: Multi-step interactive configuration sequence completes smoothly without exception",
    e2eWorkflowPassed && elements["calc-applied-tier"].textContent.includes("Tier 3"),
    `Expected smooth completion and Tier 3, got tier=${elements["calc-applied-tier"].textContent}`
  );

  // Restore globals
  if (originalDoc !== undefined) global.document = originalDoc; else delete global.document;
  if (originalWin !== undefined) global.window = originalWin; else delete global.window;

  console.log("=================================================");
  console.log(`TEST RUN FINISHED: ${results.passed}/${results.total} PASSED`);
  if (results.failed > 0) {
    console.error(`FAILED: ${results.failed} tests`);
    if (typeof process !== 'undefined') {
      try {
        const fs = require('fs');
        const path = require('path');
        const failureList = results.failures.map(f => `- FAIL: ${f.testName}\n  Reason: ${f.message}`).join('\n\n');
        const report = `# Failed pricing Tests Report

Generated by pricing-tests.js test runner for test suite \`pricing\`.

## Summary
- **Module**: \`pricing\`
- **Failures in Module**: ${results.failed}
- **Total Tests**: ${results.total}
- **Action Required**: The developer or AGY tool must resolve the failing assertions below.

---

## Failure Details

\`\`\`text
${failureList}
\`\`\`

---

## Instructions for Developer

1. Inspect the failing test assertion(s) in \`pricing-tests.js\`.
2. Do **NOT** modify or revert any files under \`data/\`. The \`data/\` directory is the immutable authoritative specification.
3. Update the matching calculations or assertions to align with authoritative data.
4. Maintain zero emojis and zero em-dashes across all files.
5. Run \`node pricing-tests.js\` to confirm the test suite is 100% green.

---

## Ready-to-Run AGY Tool Prompt

To automatically resolve these test failures using the Google Antigravity CLI, run:

\`\`\`bash
agy --print "Fix the unit test failures reported in failed_pricing_tests.report.md"
\`\`\`

Or pass this prompt directly into your active Antigravity session:

> Fix the unit test failures reported in failed_pricing_tests.report.md
`;
        fs.writeFileSync(path.join(__dirname, 'failed_pricing_tests.report.md'), report, 'utf8');
        fs.writeFileSync(path.join(__dirname, 'failed_pricing-tests_tests.report.md'), report, 'utf8');
      } catch (e) { }
      process.exit(1);
    }
  } else {
    console.log("ALL TESTS COMPLETED SUCCESSFULLY WITH ZERO ERRORS.");
    if (typeof process !== 'undefined') {
      try {
        const fs = require('fs');
        const path = require('path');
        const r1 = path.join(__dirname, 'failed_pricing_tests.report.md');
        const r2 = path.join(__dirname, 'failed_pricing-tests_tests.report.md');
        if (fs.existsSync(r1)) fs.unlinkSync(r1);
        if (fs.existsSync(r2)) fs.unlinkSync(r2);
      } catch (e) { }
    }
  }

  return results;
}

if (typeof require !== 'undefined' && require.main === module) {
  runTestSuite();
}

if (typeof window !== 'undefined') {
  window.runPricingTestSuite = runTestSuite;
}
