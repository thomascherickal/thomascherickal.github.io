/**
 * test_tier_calculators.js - Comprehensive Tests for Tier-Specific Calculators (Tiers 1-6)
 *
 * Verifies that pricing-calculator/pricing-calculator-tier-[1-6].js match
 * the authoritative pricing model:
 * - Business & Personal Flagship: 10 seats included, +$100/extra seat (PPP-scaled), max 20 seats.
 * - Business & Personal Service Picks: 1 seat included, +$100/extra seat (PPP-scaled), max 20 seats.
 * - Custom negotiated scope and boundary clamping.
 *
 * Strict Repo Compliance: Zero emojis, zero em-dashes.
 */

const path = require('path');

function run(harness) {
  const { assert, setSection } = harness;

  setSection("Tier-Specific Calculators (Tiers 1-6) - Business & Personal Parity");

  const TIER_EXPECTATIONS = [
    { id: 1, mult: 1.0, discount: 0 },
    { id: 2, mult: 0.75, discount: 25 },
    { id: 3, mult: 0.5, discount: 50 },
    { id: 4, mult: 0.5, discount: 50 },
    { id: 5, mult: 0.25, discount: 75 },
    { id: 6, mult: 0.1, discount: 90 }
  ];

  TIER_EXPECTATIONS.forEach(({ id, mult, discount }) => {
    const calcPath = path.resolve(__dirname, `../../pricing-calculator/pricing-calculator-tier-${id}.js`);
    const calc = require(calcPath);

    // 1. Metadata check
    assert(
      `Tier ${id} calculator exports calculatePricing and tier metadata (${Math.round(mult * 100)}% rate)`,
      typeof calc.calculatePricing === "function" &&
      calc.tier &&
      calc.tier.id === id &&
      calc.tier.multiplier === mult &&
      calc.tier.discountPercent === discount,
      `Tier ${id} metadata mismatch`
    );

    // 2. Business Flagship (10 seats included)
    const bizFlag10 = calc.calculatePricing({
      clientMode: "business",
      packageType: "pkg-business",
      participants: 10
    });
    const expectedFlagBase = Math.round(1000 * mult);
    assert(
      `Tier ${id} Business Flagship (10 seats) = $${expectedFlagBase} USD with 0 extra seats`,
      bizFlag10.finalPrice === expectedFlagBase &&
      bizFlag10.participants === 10 &&
      bizFlag10.extraParticipants === 0,
      `Expected $${expectedFlagBase}, got $${bizFlag10.finalPrice}`
    );

    // 3. Personal Flagship (10 seats included - identical to Business Flagship)
    const persFlag10 = calc.calculatePricing({
      clientMode: "individual",
      packageType: "pkg-personal",
      participants: 10
    });
    assert(
      `Tier ${id} Personal Flagship (10 seats) = $${expectedFlagBase} USD with 0 extra seats`,
      persFlag10.finalPrice === expectedFlagBase &&
      persFlag10.participants === 10 &&
      persFlag10.extraParticipants === 0,
      `Expected $${expectedFlagBase}, got $${persFlag10.finalPrice}`
    );

    // 4. Personal Flagship extra seats (15 seats = 5 extra seats @ $100 * mult)
    const persFlag15 = calc.calculatePricing({
      clientMode: "individual",
      packageType: "pkg-personal",
      participants: 15
    });
    const expectedExtra5 = Math.round(5 * 100 * mult);
    assert(
      `Tier ${id} Personal Flagship (15 seats) = $${expectedFlagBase + expectedExtra5} USD (5 extra seats PPP-scaled)`,
      persFlag15.participants === 15 &&
      persFlag15.extraParticipants === 5 &&
      persFlag15.extraHeadRate === expectedExtra5 &&
      persFlag15.finalPrice === expectedFlagBase + expectedExtra5,
      `Expected $${expectedFlagBase + expectedExtra5}, got $${persFlag15.finalPrice}`
    );

    // 5. Max 20 seats cap enforcement on Personal Flagship (30 requested -> clamped to 20)
    const persFlagCap = calc.calculatePricing({
      clientMode: "individual",
      packageType: "pkg-personal",
      participants: 30
    });
    const expectedExtra10 = Math.round(10 * 100 * mult);
    assert(
      `Tier ${id} Personal Flagship clamps 30 participants to 20 max (10 extra seats)`,
      persFlagCap.participants === 20 &&
      persFlagCap.extraParticipants === 10 &&
      persFlagCap.finalPrice === expectedFlagBase + expectedExtra10,
      `Expected 20 participants / $${expectedFlagBase + expectedExtra10}, got ${persFlagCap.participants} / $${persFlagCap.finalPrice}`
    );

    // 6. Service Pick (10 seats included, 13 seats = 3 extra seats PPP-scaled)
    const svcPick13 = calc.calculatePricing({
      clientMode: "individual",
      packageType: "pick-services",
      selectedServices: ["service-generative-ai-transformation"],
      participants: 13
    });
    const expectedSvcBase = Math.round(100 * mult);
    const expectedSvcExtra3 = Math.round(3 * 100 * mult);
    assert(
      `Tier ${id} Personal Service Pick (1 service, 13 seats) = $${expectedSvcBase + expectedSvcExtra3} USD (3 extra seats)`,
      svcPick13.participants === 13 &&
      svcPick13.extraParticipants === 3 &&
      svcPick13.finalPrice === expectedSvcBase + expectedSvcExtra3,
      `Expected $${expectedSvcBase + expectedSvcExtra3}, got $${svcPick13.finalPrice}`
    );
  });
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { run };
}
