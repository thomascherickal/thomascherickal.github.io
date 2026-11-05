/**
 * runner.js - Modular JavaScript Test Suite Runner
 * Orchestrates and executes all modular test files under tests/js/.
 * Strict Repo Compliance: Zero emojis, zero em-dashes.
 */

const { createHarness } = require('./test_helpers.js');
const testDataIntegrity = require('./test_data_integrity.js');
const testZeroSelection = require('./test_zero_selection.js');
const testIndividualServices = require('./test_individual_services.js');
const testBusinessServices = require('./test_business_services.js');
const testExtraParticipants = require('./test_extra_participants.js');
const testFlagshipPackages = require('./test_flagship_packages.js');
const testCustomPackage = require('./test_custom_package.js');
const testEdgeCases = require('./test_edge_cases.js');
const testDomController = require('./test_dom_controller.js');

const TEST_MODULES = [
  { name: "test_data_integrity", module: testDataIntegrity },
  { name: "test_zero_selection", module: testZeroSelection },
  { name: "test_individual_services", module: testIndividualServices },
  { name: "test_business_services", module: testBusinessServices },
  { name: "test_extra_participants", module: testExtraParticipants },
  { name: "test_flagship_packages", module: testFlagshipPackages },
  { name: "test_custom_package", module: testCustomPackage },
  { name: "test_edge_cases", module: testEdgeCases },
  { name: "test_dom_controller", module: testDomController }
];

function runAllModules(PricingCalculator, constants) {
  const harness = createHarness();
  const startTime = Date.now();

  for (const { module } of TEST_MODULES) {
    module.run(harness, PricingCalculator, constants);
  }

  harness.printSummary("JAVASCRIPT UNIT TEST SUITE (pricing-tests.js)", startTime);

  if (typeof process !== 'undefined') {
    try {
      const fs = require('fs');
      const path = require('path');
      const cachePath = path.join(__dirname, '..', '.js_test_reports_cache.json');
      const cacheData = {
        timestamp: new Date().toISOString(),
        total: harness.results.total,
        passed: harness.results.passed,
        failed: harness.results.failed,
        elapsed_seconds: ((Date.now() - startTime) / 1000).toFixed(2),
        status: harness.results.failed === 0 ? "PASSED" : "FAILED",
        sections: harness.sections
      };
      fs.writeFileSync(cachePath, JSON.stringify(cacheData, null, 2), 'utf8');
    } catch (e) { }
  }

  if (harness.results.failed > 0) {
    harness.handleFailures("pricing");
  } else {
    harness.handleFailures("pricing"); // Cleans stale reports
  }

  return harness.results;
}

module.exports = {
  TEST_MODULES,
  runAllModules
};

if (typeof require !== 'undefined' && require.main === module) {
  const PricingCalculator = require('../../pricing-calculator.js');
  const { constants } = require('../../pricing-tests.js');
  runAllModules(PricingCalculator, constants);
}

