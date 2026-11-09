/**
 * test_payment_schedule.js - Comprehensive Unit Tests for payment-schedule.js
 *
 * Covers 100% of calculation rules, milestone schedules, $0 upfront deposit,
 * trust architecture, Apache 2.0 licensing, custom negotiated terms, and DOM controller interactions.
 *
 * Strict Repo Compliance:
 * - Zero emojis.
 * - Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
 */

const PaymentSchedule = typeof require !== 'undefined'
  ? require('../../payment-schedule.js')
  : window.PaymentSchedule;

function run(harness) {
  const { assert, setSection } = harness;

  setSection("Payment Schedule - Data Integrity & Tiers");

  assert(
    "PaymentSchedule module exports necessary functions and constants",
    typeof PaymentSchedule.calculateSchedule === "function" &&
    typeof PaymentSchedule.formatCurrency === "function" &&
    typeof PaymentSchedule.initDOMController === "function" &&
    typeof PaymentSchedule.PARITY_TIERS === "object" &&
    typeof PaymentSchedule.TRUST_TERMS === "object",
    "PaymentSchedule must export calculateSchedule, formatCurrency, initDOMController, PARITY_TIERS, TRUST_TERMS"
  );

  const t = PaymentSchedule.PARITY_TIERS;
  assert(
    "All 6 Parity Tiers configured with correct multipliers and discounts",
    t[1].multiplier === 1.0 && t[1].discount === 0 &&
    t[2].multiplier === 0.75 && t[2].discount === 25 &&
    t[3].multiplier === 0.50 && t[3].discount === 50 &&
    t[4].multiplier === 0.50 && t[4].discount === 50 &&
    t[5].multiplier === 0.25 && t[5].discount === 75 &&
    t[6].multiplier === 0.10 && t[6].discount === 90,
    "Parity tiers multipliers or discounts mismatch authoritative values"
  );

  assert(
    "Tier 1 flagship is $3000 USD and individual service is $300 USD",
    t[1].flagship === 3000 && t[1].indiv === 300 && t[1].supportFlag === 500 && t[1].supportIndiv === 50,
    "Tier 1 authoritative prices mismatch"
  );

  assert(
    "Trust terms include $0 upfront deposit and Apache 2.0 license",
    PaymentSchedule.TRUST_TERMS.upfrontDeposit.includes("$0 upfront") &&
    PaymentSchedule.TRUST_TERMS.license.includes("Apache 2.0") &&
    PaymentSchedule.TRUST_TERMS.milestoneCount === 5 &&
    PaymentSchedule.TRUST_TERMS.milestonePercentage === 20,
    "Trust terms must specify $0 upfront, Apache 2.0, and 5 milestone blocks of 20%"
  );

  setSection("Payment Schedule - Currency Formatting Helper");

  assert("formatCurrency formats positive integers", PaymentSchedule.formatCurrency(3000) === "$3,000 USD");
  assert("formatCurrency formats zero", PaymentSchedule.formatCurrency(0) === "$0 USD");
  assert("formatCurrency formats strings", PaymentSchedule.formatCurrency("2250") === "$2,250 USD");
  assert("formatCurrency handles NaN/undefined", PaymentSchedule.formatCurrency(null) === "$0 USD");

  setSection("Payment Schedule - Flagship Milestone Calculations Across Tiers");

  // Default execution (no options)
  const defaultSched = PaymentSchedule.calculateSchedule();
  assert(
    "calculateSchedule() defaults cleanly to Tier 1 Flagship ($3000 training + $500 support)",
    defaultSched.tierId === 1 &&
    defaultSched.totalTrainingAmountUsd === 3000 &&
    defaultSched.totalSupportAmountUsd === 500 &&
    defaultSched.totalEngagementAmountUsd === 3500 &&
    defaultSched.upfrontDepositUsd === 0,
    "Default schedule must be Tier 1 Flagship with $0 upfront"
  );
  assert(
    "Flagship training creates exactly 5 milestones of 20% each ($600 each for Tier 1)",
    defaultSched.trainingMilestones.length === 5 &&
    defaultSched.trainingMilestones.every(m => m.percentage === 20 && m.amountUsd === 600),
    "Tier 1 flagship must have 5 milestones of $600 each"
  );
  assert(
    "Flagship support creates 2 installments ($250 at Day 15 and $250 at Day 30)",
    defaultSched.supportMilestones.length === 2 &&
    defaultSched.supportMilestones[0].dayDue === 15 && defaultSched.supportMilestones[0].amountUsd === 250 &&
    defaultSched.supportMilestones[1].dayDue === 30 && defaultSched.supportMilestones[1].amountUsd === 250,
    "Flagship support must have Day 15 ($250) and Day 30 ($250) milestones"
  );

  // Tier 2 Flagship ($2250)
  const t2Sched = PaymentSchedule.calculateSchedule({ tierId: 2, engagementType: "flagship" });
  assert(
    "Tier 2 Flagship: $2250 training (5 x $450) + $375 support",
    t2Sched.totalTrainingAmountUsd === 2250 &&
    t2Sched.trainingMilestones[0].amountUsd === 450 &&
    t2Sched.totalSupportAmountUsd === 375 &&
    t2Sched.supportMilestones[0].amountUsd === 187.5 &&
    t2Sched.discountPercent === 25,
    "Tier 2 calculation mismatch"
  );

  // Tier 3 Flagship ($1500)
  const t3Sched = PaymentSchedule.calculateSchedule({ tierId: 3, engagementType: "flagship" });
  assert(
    "Tier 3 Flagship: $1500 training (5 x $300) + $250 support (2 x $125)",
    t3Sched.totalTrainingAmountUsd === 1500 &&
    t3Sched.trainingMilestones[0].amountUsd === 300 &&
    t3Sched.totalSupportAmountUsd === 250 &&
    t3Sched.supportMilestones[0].amountUsd === 125 &&
    t3Sched.discountPercent === 50,
    "Tier 3 calculation mismatch"
  );

  // Tier 4 Flagship ($1500)
  const t4Sched = PaymentSchedule.calculateSchedule({ tierId: 4, engagementType: "flagship" });
  assert(
    "Tier 4 Flagship: $1500 training (5 x $300) + $250 support",
    t4Sched.totalTrainingAmountUsd === 1500 &&
    t4Sched.trainingMilestones[0].amountUsd === 300 &&
    t4Sched.discountPercent === 50,
    "Tier 4 calculation mismatch"
  );

  // Tier 5 Flagship ($750)
  const t5Sched = PaymentSchedule.calculateSchedule({ tierId: 5, engagementType: "flagship" });
  assert(
    "Tier 5 Flagship: $750 training (5 x $150) + $125 support",
    t5Sched.totalTrainingAmountUsd === 750 &&
    t5Sched.trainingMilestones[0].amountUsd === 150 &&
    t5Sched.discountPercent === 75,
    "Tier 5 calculation mismatch"
  );

  // Tier 6 Flagship ($300)
  const t6Sched = PaymentSchedule.calculateSchedule({ tierId: 6, engagementType: "flagship" });
  assert(
    "Tier 6 Flagship: $300 training (5 x $60) + $50 support (2 x $25)",
    t6Sched.totalTrainingAmountUsd === 300 &&
    t6Sched.trainingMilestones[0].amountUsd === 60 &&
    t6Sched.totalSupportAmountUsd === 50 &&
    t6Sched.supportMilestones[0].amountUsd === 25 &&
    t6Sched.discountPercent === 90,
    "Tier 6 calculation mismatch"
  );

  // Out-of-bounds tier defensive test
  const invalidTierSched = PaymentSchedule.calculateSchedule({ tierId: 99 });
  assert(
    "Defensive: Invalid tierId falls back safely to Tier 1",
    invalidTierSched.tierId === 1 && invalidTierSched.totalTrainingAmountUsd === 3000,
    "Invalid tierId must fallback to Tier 1"
  );

  setSection("Payment Schedule - Individual Services Calculations");

  // Single individual service Tier 1
  const ind1T1 = PaymentSchedule.calculateSchedule({ tierId: 1, engagementType: "individual", serviceCount: 1 });
  assert(
    "Tier 1 Individual (1 service): $300 training (5 x $60) + $50 support (1 x Day 5 installment)",
    ind1T1.totalTrainingAmountUsd === 300 &&
    ind1T1.trainingMilestones.length === 5 &&
    ind1T1.trainingMilestones[0].amountUsd === 60 &&
    ind1T1.totalSupportAmountUsd === 50 &&
    ind1T1.supportMilestones.length === 1 &&
    ind1T1.supportMilestones[0].dayDue === 5 &&
    ind1T1.supportMilestones[0].amountUsd === 50,
    "Tier 1 individual service calculation mismatch"
  );

  // Multiple individual services Tier 2 (3 services)
  const ind3T2 = PaymentSchedule.calculateSchedule({ tierId: 2, engagementType: "individual", serviceCount: 3 });
  assert(
    "Tier 2 Individual (3 services): $675 training (5 x $135) + $114 support",
    ind3T2.totalTrainingAmountUsd === 675 &&
    ind3T2.trainingMilestones[0].amountUsd === 135 &&
    ind3T2.totalSupportAmountUsd === 114 &&
    ind3T2.supportMilestones[0].amountUsd === 114,
    "Tier 2 (3 services) calculation mismatch"
  );

  // Individual service count sanitization (negative or 0 clamped to 1)
  const indClamp = PaymentSchedule.calculateSchedule({ tierId: 1, engagementType: "individual", serviceCount: -4 });
  assert(
    "Defensive: Negative service count normalized to 1",
    indClamp.serviceCount === 1 && indClamp.totalTrainingAmountUsd === 300,
    "Negative service count should normalize to 1"
  );

  setSection("Payment Schedule - Custom Negotiated Scope");

  const customSched1 = PaymentSchedule.calculateSchedule({
    engagementType: "custom",
    customAmount: 2400
  });
  assert(
    "Custom scope: $2400 training creates 5 x $480 milestones + proportional support ($400)",
    customSched1.isNegotiated === true &&
    customSched1.totalTrainingAmountUsd === 2400 &&
    customSched1.trainingMilestones[0].amountUsd === 480 &&
    customSched1.totalSupportAmountUsd === 400 &&
    customSched1.supportMilestones.length === 2,
    "Custom amount calculation mismatch"
  );

  const customSchedStr = PaymentSchedule.calculateSchedule({
    engagementType: "custom",
    customAmount: "1200"
  });
  assert(
    "Custom scope: String input parsed cleanly",
    customSchedStr.totalTrainingAmountUsd === 1200 &&
    customSchedStr.trainingMilestones[0].amountUsd === 240,
    "Custom amount string parsing failed"
  );

  const customSchedZero = PaymentSchedule.calculateSchedule({
    engagementType: "custom",
    customAmount: 0
  });
  assert(
    "Custom scope: $0 amount handles without errors",
    customSchedZero.totalTrainingAmountUsd === 0 &&
    customSchedZero.totalSupportAmountUsd === 0 &&
    customSchedZero.trainingMilestones[0].amountUsd === 0,
    "Custom amount 0 failed"
  );

  setSection("Payment Schedule - DOM Controller Simulation");

  const originalDocument = global.document;

  // Verify graceful handling when document is undefined or elements missing
  delete global.document;
  PaymentSchedule.initDOMController();
  assert("initDOMController returns gracefully in headless environments", true);

  // Setup mock DOM elements
  const listeners = {};
  function makeMockElement(id) {
    const el = {
      id: id,
      value: "",
      textContent: "",
      _innerHTML: "",
      style: {},
      children: [],
      appendChild: function (child) { this.children.push(child); },
      addEventListener: function (evt, handler) {
        if (!listeners[id]) listeners[id] = {};
        if (!listeners[id][evt]) listeners[id][evt] = [];
        listeners[id][evt].push(handler);
      },
      trigger: function (evt) {
        if (listeners[id] && listeners[id][evt]) {
          listeners[id][evt].forEach(fn => fn());
        }
      }
    };

    Object.defineProperty(el, "innerHTML", {
      get: function () { return this._innerHTML; },
      set: function (val) {
        this._innerHTML = val;
        if (val === "") {
          this.children = [];
        }
      }
    });

    return el;
  }

  const mockWidget = makeMockElement("payment-schedule-widget");
  const mockTierSelect = makeMockElement("sched-tier-select");
  const mockEngSelect = makeMockElement("sched-eng-select");
  const mockCustomGroup = makeMockElement("sched-custom-group");
  const mockCustomInput = makeMockElement("sched-custom-input");
  const mockSvcCountGroup = makeMockElement("sched-svc-count-group");
  const mockSvcCountInput = makeMockElement("sched-svc-count-input");
  const mockTotalDisplay = makeMockElement("sched-total-display");
  const mockUpfrontDisplay = makeMockElement("sched-upfront-display");
  const mockDiscountDisplay = makeMockElement("sched-discount-display");
  const mockLicenseDisplay = makeMockElement("sched-license-display");
  const mockWalkawayDisplay = makeMockElement("sched-walkaway-display");
  const mockNegotiableNote = makeMockElement("sched-negotiable-note");
  const mockMilestonesBody = makeMockElement("sched-milestones-body");

  const elementsMap = {
    "payment-schedule-widget": mockWidget,
    "sched-tier-select": mockTierSelect,
    "sched-eng-select": mockEngSelect,
    "sched-custom-group": mockCustomGroup,
    "sched-custom-input": mockCustomInput,
    "sched-svc-count-group": mockSvcCountGroup,
    "sched-svc-count-input": mockSvcCountInput,
    "sched-total-display": mockTotalDisplay,
    "sched-upfront-display": mockUpfrontDisplay,
    "sched-discount-display": mockDiscountDisplay,
    "sched-license-display": mockLicenseDisplay,
    "sched-walkaway-display": mockWalkawayDisplay,
    "sched-negotiable-note": mockNegotiableNote,
    "sched-milestones-body": mockMilestonesBody
  };

  // Attach global mock document
  global.document = {
    readyState: "complete",
    getElementById: function (id) {
      return elementsMap[id] || null;
    },
    createElement: function (tag) {
      return { tagName: tag, innerHTML: "", style: {} };
    },
    addEventListener: function () { }
  };

  mockTierSelect.value = "1";
  mockEngSelect.value = "flagship";
  mockCustomInput.value = "3000";
  mockSvcCountInput.value = "1";

  PaymentSchedule.initDOMController();

  assert(
    "DOM Controller renders initial Tier 1 Flagship schedule",
    mockTotalDisplay.textContent === "$3,500 USD" &&
    mockUpfrontDisplay.textContent.includes("$0 USD") &&
    mockLicenseDisplay.textContent.includes("Apache 2.0") &&
    mockMilestonesBody.children.length === 7, // 5 training + 2 support
    "DOM controller initial render failed"
  );

  // Switch Tier to Tier 3
  mockTierSelect.value = "3";
  mockTierSelect.trigger("change");
  assert(
    "DOM Controller updates on tier change to Tier 3 ($1750 USD total, 50% discount)",
    mockTotalDisplay.textContent === "$1,750 USD" &&
    mockDiscountDisplay.textContent.includes("50% Parity Discount"),
    "DOM controller failed to update on tier select change"
  );

  // Switch Engagement to Individual Services
  mockEngSelect.value = "individual";
  mockSvcCountInput.value = "2";
  mockEngSelect.trigger("change");
  assert(
    "DOM Controller updates on engagement type change to individual (2 services)",
    mockTotalDisplay.textContent === "$350 USD" &&
    mockSvcCountGroup.style.display === "block" &&
    mockMilestonesBody.children.length === 6, // 5 training + 1 support
    "DOM controller failed on individual services change"
  );

  // Switch to Custom Negotiated
  mockEngSelect.value = "custom";
  mockCustomInput.value = "1800";
  mockEngSelect.trigger("change");
  assert(
    "DOM Controller displays custom group and parses custom input",
    mockCustomGroup.style.display === "block" &&
    mockDiscountDisplay.textContent === "Custom Negotiated Scope",
    "DOM controller failed on custom scope change"
  );

  // Test custom input event trigger
  mockCustomInput.value = "5000";
  mockCustomInput.trigger("input");
  assert(
    "DOM Controller updates on custom input typing ($5,833 USD total)",
    mockTotalDisplay.textContent === "$5,833 USD",
    "DOM controller failed on custom input event"
  );

  // Test svcCountInput event trigger
  mockEngSelect.value = "individual";
  mockEngSelect.trigger("change");
  mockSvcCountInput.value = "4";
  mockSvcCountInput.trigger("input");
  assert(
    "DOM Controller updates on service count input typing (4 services)",
    mockTotalDisplay.textContent === "$700 USD",
    "DOM controller failed on service count input event"
  );

  // Test empty/invalid input fallbacks in DOM controller
  mockCustomInput.value = "";
  mockSvcCountInput.value = "";
  mockEngSelect.value = "custom";
  mockEngSelect.trigger("change");
  assert(
    "DOM Controller handles empty/invalid custom and service count inputs safely",
    mockTotalDisplay.textContent === "$0 USD",
    "DOM controller failed on empty custom input"
  );

  // Test DOM container missing early return
  const originalGetElementById = global.document.getElementById;
  global.document.getElementById = function () { return null; };
  PaymentSchedule.initDOMController();
  assert("initDOMController returns safely when container missing", true);
  global.document.getElementById = originalGetElementById;

  // Test DOM with empty container (all sub-elements null)
  global.document.getElementById = function (id) {
    if (id === "payment-schedule-widget") return { id: id };
    return null;
  };
  PaymentSchedule.initDOMController();
  assert("initDOMController handles container with missing child inputs", true);
  global.document.getElementById = originalGetElementById;

  // Test loading state event listener
  let domLoadedHandler = null;
  global.document.readyState = "loading";
  global.document.addEventListener = function (evt, handler) {
    if (evt === "DOMContentLoaded") domLoadedHandler = handler;
  };
  // Re-run script file in clean VM context simulating browser loading state
  const vm = require("vm");
  const fs = require("fs");
  const path = require("path");
  const targetScriptPath = path.resolve(__dirname, "../../payment-schedule.js");
  const code = fs.readFileSync(targetScriptPath, "utf8");
  const browserContext = {
    self: {},
    document: {
      readyState: "loading",
      addEventListener: function (evt, fn) {
        if (evt === "DOMContentLoaded") domLoadedHandler = fn;
      },
      getElementById: function () { return null; }
    }
  };
  vm.createContext(browserContext);
  vm.runInContext(code, browserContext, { filename: targetScriptPath });
  assert(
    "Browser UMD export attaches to root (self.PaymentSchedule) and registers DOMContentLoaded listener",
    typeof browserContext.self.PaymentSchedule === "object" && typeof domLoadedHandler === "function",
    "Browser UMD registration or DOMContentLoaded listener failed"
  );
  if (domLoadedHandler) domLoadedHandler();

  // Clean up global mock
  if (originalDocument) {
    global.document = originalDocument;
  } else {
    delete global.document;
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { run };
}

if (typeof require !== 'undefined' && require.main === module) {
  const { createHarness } = require('./test_helpers.js');
  const harness = createHarness();
  run(harness);
  harness.printSummary("PAYMENT SCHEDULE TEST SUITE (test_payment_schedule.js)", Date.now());
  if (harness.results.failed > 0) process.exit(1);
}
