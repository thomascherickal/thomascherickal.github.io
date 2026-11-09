/**
 * pricing-calculator-tier-1-tests.js
 * Unit Test Suite for Tier 1 Dedicated Pricing Calculator.
 * Runs under Node.js with ZERO external dependencies.
 *
 * Strict Repo Compliance:
 * - Zero emojis.
 * - Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
 */

const assert = require('assert');
const Calculator = require('./pricing-calculator-tier-1.js');

const tier = Calculator.tier;
console.log(`Running tests for Tier 1: ${tier.shortName} (Multiplier: ${tier.multiplier}x, Discount: ${tier.discountPercent}%)...`);

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  [PASS] ${name}`);
  } catch (err) {
    failed++;
    console.error(`  [FAIL] ${name}: ${err.message}`);
  }
}

// 1. Metadata tests
test("Metadata verification", () => {
  assert.strictEqual(tier.id, 1);
  assert.strictEqual(typeof tier.multiplier, "number");
  assert.strictEqual(typeof tier.discountPercent, "number");
  assert.ok(tier.badgeText.length > 0);
});

// 2. Business Flagship calculation
test("Business Flagship calculation (10 seats baseline)", () => {
  const res = Calculator.calculatePricing({
    clientMode: "business",
    packageType: "pkg-business",
    participants: 10
  });
  const expectedBase = Math.round(3000 * tier.multiplier);
  const expectedSavings = Math.round(3000 * (tier.discountPercent / 100));
  assert.strictEqual(res.baseTierPrice, expectedBase);
  assert.strictEqual(res.extraHeadRate, 0);
  assert.strictEqual(res.finalPrice, expectedBase);
  assert.strictEqual(res.totalSavings, expectedSavings);
});

// 3. Business Flagship with extra participants
test("Business Flagship with extra participants (15 seats)", () => {
  const res = Calculator.calculatePricing({
    clientMode: "business",
    packageType: "pkg-business",
    participants: 15
  });
  const expectedExtra = Math.round(5 * 300 * tier.multiplier);
  const expectedTotal = Math.round(3000 * tier.multiplier) + expectedExtra;
  assert.strictEqual(res.extraParticipants, 5);
  assert.strictEqual(res.extraHeadRate, expectedExtra);
  assert.strictEqual(res.finalPrice, expectedTotal);
});

// 4. Personal Flagship calculation
test("Personal Flagship calculation (strictly 1 person)", () => {
  const res = Calculator.calculatePricing({
    clientMode: "individual",
    packageType: "pkg-personal",
    participants: 10
  });
  const expectedBase = Math.round(3000 * tier.multiplier);
  assert.strictEqual(res.participants, 1);
  assert.strictEqual(res.extraParticipants, 0);
  assert.strictEqual(res.finalPrice, expectedBase);
});

// 5. Individual pick-services
test("Pick individual services (3 services)", () => {
  const res = Calculator.calculatePricing({
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: ["s1", "s2", "s3"]
  });
  const expectedBase = Math.round(900 * tier.multiplier);
  assert.strictEqual(res.serviceCount, 3);
  assert.strictEqual(res.basePrice, 900);
  assert.strictEqual(res.finalPrice, expectedBase);
});

// 6. Nothing selected (0 services)
test("Zero services selected", () => {
  const res = Calculator.calculatePricing({
    clientMode: "individual",
    packageType: "pick-services",
    selectedServices: []
  });
  assert.strictEqual(res.basePrice, 0);
  assert.strictEqual(res.finalPrice, 0);
  assert.strictEqual(res.totalSavings, 0);
});

// 7. Custom amount package
test("Custom amount calculation", () => {
  const res = Calculator.calculatePricing({
    clientMode: "business",
    packageType: "custom",
    customBase: 5000,
    participants: 10
  });
  const expectedBase = Math.round(5000 * tier.multiplier);
  assert.strictEqual(res.basePrice, 5000);
  assert.strictEqual(res.finalPrice, expectedBase);
});

// 8. Flagship cheaper nudge check
test("Flagship cheaper recommendation when >= 10 services picked", () => {
  const tenServices = Array(11).fill("service");
  const res = Calculator.calculatePricing({
    clientMode: "business",
    packageType: "pick-services",
    selectedServices: tenServices
  });
  assert.strictEqual(res.flagshipCheaper, true);
  assert.strictEqual(res.flagshipPrice, 3000);
});

console.log(`Results for Tier 1: ${passed} passed, ${failed} failed.`);
if (failed > 0) process.exit(1);
