/**
 * test_data_integrity.js - Data Integrity & Metadata Tests
 * Strict Repo Compliance: Zero emojis, zero em-dashes.
 */

function run(harness, PricingCalculator) {
  const { assert, setSection } = harness;
  setSection("Data Integrity & Metadata Tests");

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
    "ALL_SERVICES_DATA contains all 14 services",
    PricingCalculator.ALL_SERVICES_DATA.length === 14,
    `Expected 14 services, found ${PricingCalculator.ALL_SERVICES_DATA.length}`
  );

  const uniqueSlugs = new Set(PricingCalculator.ALL_SERVICES_DATA.map(s => s.slug));
  assert(
    "ALL_SERVICES_DATA has 14 unique slugs without duplicates",
    uniqueSlugs.size === 14,
    `Expected 14 unique slugs, found ${uniqueSlugs.size}`
  );

  // Verify distinct services matching business-services/ and personal-services/ directories
  const fs = require("fs");
  const path = require("path");
  const rootDir = path.resolve(__dirname, "../../");
  const bFiles = fs.readdirSync(path.join(rootDir, "business-services"))
    .filter(f => f.endsWith(".html") && !f.startsWith("all-"));
  const pFiles = fs.readdirSync(path.join(rootDir, "personal-services"))
    .filter(f => f.endsWith(".html") && !f.startsWith("all-"));

  const bSlugs = bFiles.map(f => "service-" + f.replace("business-service-", "").replace(".html", ""));
  const pSlugs = pFiles.map(f => "service-" + f.replace("personal-service-", "").replace(".html", ""));
  const expectedSlugs = Array.from(new Set([...bSlugs, ...pSlugs])).sort();

  assert(
    "ALL_SERVICES_DATA matches all distinct services in business-services/ and personal-services/",
    expectedSlugs.length === 14 &&
    expectedSlugs.every(slug => uniqueSlugs.has(slug)),
    `Expected distinct slugs ${JSON.stringify(expectedSlugs)} to match calculator slugs ${JSON.stringify(Array.from(uniqueSlugs).sort())}`
  );

  // Verify all pricing calculator scripts export all 14 distinct services
  const tierCalcs = [
    { name: "tier-1", mod: require("../../pricing-calculator/pricing-calculator-tier-1.js") },
    { name: "tier-2", mod: require("../../pricing-calculator/pricing-calculator-tier-2.js") },
    { name: "tier-3", mod: require("../../pricing-calculator/pricing-calculator-tier-3.js") },
    { name: "tier-4", mod: require("../../pricing-calculator/pricing-calculator-tier-4.js") },
    { name: "tier-5", mod: require("../../pricing-calculator/pricing-calculator-tier-5.js") },
    { name: "tier-6", mod: require("../../pricing-calculator/pricing-calculator-tier-6.js") }
  ];

  tierCalcs.forEach(({ name, mod }) => {
    assert(
      `PricingCalculatorTier (${name}) exports all 14 distinct services`,
      Array.isArray(mod.ALL_SERVICES_DATA) &&
      mod.ALL_SERVICES_DATA.length === 14 &&
      new Set(mod.ALL_SERVICES_DATA.map(s => s.slug)).size === 14 &&
      expectedSlugs.every(slug => mod.ALL_SERVICES_DATA.some(s => s.slug === slug)),
      `Tier calculator ${name} does not match 14 distinct services`
    );
  });

  // Verify all 7 pricing pages contain calculator markup with services-checkbox-grid and All 14 button
  const pricingPages = [
    "pricing.html",
    "pricing/pricing-tier-1-usa-china-eu-100.html",
    "pricing/pricing-tier-2-brazil-malaysia-75.html",
    "pricing/pricing-tier-3-india-egypt-50.html",
    "pricing/pricing-tier-4-tech-startups-50.html",
    "pricing/pricing-tier-5-colleges-non-profits-25.html",
    "pricing/pricing-tier-6-crisis-affected-10.html"
  ];

  pricingPages.forEach(relPath => {
    const filePath = path.join(rootDir, relPath);
    assert(
      `Pricing page exists: ${relPath}`,
      fs.existsSync(filePath),
      `File ${filePath} not found`
    );
    const htmlContent = fs.readFileSync(filePath, "utf8");
    assert(
      `Pricing page ${relPath} contains services-checkbox-grid container`,
      htmlContent.includes('id="services-checkbox-grid"'),
      `${relPath} is missing id="services-checkbox-grid"`
    );
    assert(
      `Pricing page ${relPath} contains Select All 14 services button`,
      /id="btn-select-all-services"[^>]*>[\s\S]*?All\s+14[\s\S]*?<\/button>/.test(htmlContent),
      `${relPath} is missing or has incorrect label for btn-select-all-services`
    );
  });
}

module.exports = { run };

if (typeof require !== 'undefined' && require.main === module) {
  const { createHarness } = require('./test_helpers.js');
  const PricingCalculator = require('../../pricing-calculator.js');
  const harness = createHarness();
  const startTime = Date.now();
  run(harness, PricingCalculator);
  harness.printSummary("JAVASCRIPT UNIT TEST (test_data_integrity.js)", startTime);
  if (harness.results.failed > 0) {
    harness.handleFailures("test_data_integrity");
  }
}

