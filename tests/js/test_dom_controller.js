/**
 * test_dom_controller.js - DOM Controller Simulated Live-Update Tests (pricing.html Pane)
 * Verifies that the price listed changes as soon as a change is made in ANY UI element.
 * Strict Repo Compliance: Zero emojis, zero em-dashes.
 */

const { setupMockDOM } = require('./test_helpers.js');

function run(harness, PricingCalculator, constants) {
  const { assert, setSection } = harness;
  const B_FLAG_PRICE = constants ? constants.B_FLAG_PRICE : 3000;
  const P_FLAG_PRICE = constants ? constants.P_FLAG_PRICE : 3000;
  const B_IND_PRICE = constants ? constants.B_IND_PRICE : 300;
  const B_EXTRA_SEAT = constants ? constants.B_EXTRA_SEAT : 300;

  setSection("DOM Controller Simulated Live-Update Tests");

  const { mockDocument, elements, tierBtns, serviceCards, getCheckboxes } = setupMockDOM(PricingCalculator);
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
    `UI Select All: Clicking All 16 immediately checks all services and updates price to $${(16 * B_IND_PRICE).toLocaleString()} USD`,
    elements["calc-display-price"].textContent === `$${(16 * B_IND_PRICE).toLocaleString()} USD` &&
    elements["selected-service-count-label"].textContent.includes("16 Services Selected"),
    `Expected $${(16 * B_IND_PRICE).toLocaleString()} USD, got ${elements["calc-display-price"].textContent}`
  );

  // 7. Participants Input Change: Extra head scaling (+5 participants over 10 included = 15)
  elements["calc-participants"].value = "15";
  elements["calc-participants"].trigger("input");
  const ui15Price = (16 * B_IND_PRICE) + (5 * B_EXTRA_SEAT);
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

  // 16. Edge Case & Crash Prevention Regression Tests
  let noErrorThrown = true;
  try {
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

  let e2eWorkflowPassed = true;
  try {
    elements["btn-mode-individual"].trigger("click");
    elements["calc-package"].value = "pick-services";
    elements["calc-package"].trigger("change");
    const cbs = getCheckboxes();
    cbs[0].checked = true;
    cbs[1].checked = true;
    cbs[2].checked = true;
    cbs[0].trigger("change");
    elements["calc-country"].selectedIndex = 3;
    elements["calc-country"].selectedOptions = [elements["calc-country"].options[3]];
    elements["calc-country"].trigger("change");
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
}

module.exports = { run };

if (typeof require !== 'undefined' && require.main === module) {
  const { createHarness } = require('./test_helpers.js');
  const PricingCalculator = require('../../pricing-calculator.js');
  const { constants } = require('../../pricing-tests.js');
  const harness = createHarness();
  const startTime = Date.now();
  run(harness, PricingCalculator, constants);
  harness.printSummary("JAVASCRIPT UNIT TEST (test_dom_controller.js)", startTime);
  if (harness.results.failed > 0) {
    harness.handleFailures("test_dom_controller");
  }
}

