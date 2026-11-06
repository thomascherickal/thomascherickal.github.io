/**
 * pricing-tests.js
 * Comprehensive Unit and Functional Test Suite for Pricing Calculator & Dynamic UI Pane.
 * Modular test runner delegating to tests/js/ modular test suites.
 * Runs under Node.js or browser console with ZERO external dependencies.
 *
 * Strict Repo Compliance:
 * - Zero emojis.
 * - Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
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

const constants = {
  B_FLAG_PRICE,
  P_FLAG_PRICE,
  B_IND_PRICE,
  B_EXTRA_SEAT,
  B_INCLUDED_SEATS
};

function runTestSuite() {
  if (typeof require !== 'undefined') {
    const { runAllModules } = require('./tests/js/runner.js');
    return runAllModules(PricingCalculator, constants);
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    runTestSuite,
    constants
  };
}

if (typeof require !== 'undefined' && require.main === module) {
  const p = runTestSuite();
  if (p && typeof p.then === 'function') {
    p.then(res => {
      if (res && res.failed > 0) process.exit(1);
    });
  }
}

if (typeof window !== 'undefined') {
  window.runPricingTestSuite = runTestSuite;
  window.pricingConstants = constants;
}
