/**
 * payment-schedule.js
 * Comprehensive Payment Schedule & Trust Architecture Widget.
 *
 * Features:
 * 1. 10 x 10% Milestone Schedule Calculations across all 6 Parity Tiers.
 * 2. Dedicated Support 100% included at zero charge ($0 USD) with no separate support payment rows.
 * 3. Strictly $0 Upfront Deposit validation and guarantees.
 * 4. Custom Special Cases calculation with negotiable rates for under-resourced ventures.
 * 5. Full Commercial Re-use Licensing (Apache 2.0) terms across all tiers.
 * 6. Dynamic DOM controller for interactive tier, engagement, and custom budget switching.
 *
 * Strict Repo Compliance:
 * - Zero emojis.
 * - Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.PaymentSchedule = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {

  const PARITY_TIERS = {
    1: { id: 1, name: "Tier 1: High-Income Economies (USA, UK, EU)", multiplier: 1.0, discount: 0, flagship: 1000, indiv: 100, supportFlag: 0, supportIndiv: 0 },
    2: { id: 2, name: "Tier 2: Upper-Middle-Income (Brazil, Malaysia, Mexico)", multiplier: 0.75, discount: 25, flagship: 750, indiv: 75, supportFlag: 0, supportIndiv: 0 },
    3: { id: 3, name: "Tier 3: Lower-Middle & Low-Income (India, Egypt, Kenya)", multiplier: 0.5, discount: 50, flagship: 500, indiv: 50, supportFlag: 0, supportIndiv: 0 },
    4: { id: 4, name: "Tier 4: Early-Stage Tech Startups & Seed Ventures", multiplier: 0.5, discount: 50, flagship: 500, indiv: 50, supportFlag: 0, supportIndiv: 0 },
    5: { id: 5, name: "Tier 5: Colleges, Universities & Non-Profits", multiplier: 0.25, discount: 75, flagship: 250, indiv: 25, supportFlag: 0, supportIndiv: 0 },
    6: { id: 6, name: "Tier 6: Crisis-Affected Regions & Subsidized Relief", multiplier: 0.1, discount: 90, flagship: 100, indiv: 10, supportFlag: 0, supportIndiv: 0 }
  };

  const TRUST_TERMS = {
    upfrontDeposit: "$0 upfront (strictly zero advance payment required before commencing technical discovery and instruction)",
    milestoneCount: 10,
    milestonePercentage: 10,
    walkawayGuarantee: "Walk away owing nothing more than already paid if not satisfied after any completed milestone training phase, including the first.",
    license: "Apache 2.0 license. All training materials, guides, architectures, and client code are transferred with full commercial re-use, internal deployment, resale, and relicensing rights with attribution.",
    negotiability: "All pricing is negotiable for special cases, non-profits, academic research groups, and crisis-affected regions."
  };

  function formatCurrency(amount) {
    const num = Math.round(Number(amount) || 0);
    return "$" + num.toLocaleString() + " USD";
  }

  /**
   * Calculates detailed payment schedule milestones for an engagement.
   * @param {Object} [options]
   * @param {number} [options.tierId=1] 1 to 6
   * @param {string} [options.engagementType="flagship"] "flagship", "individual", or "custom"
   * @param {number} [options.serviceCount=1] Number of individual services
   * @param {number|string} [options.customAmount=0] Custom negotiated amount in USD
   * @returns {Object} Calculated schedule details
   */
  function calculateSchedule(options) {
    const opts = options || {};
    const rawTierId = parseInt(opts.tierId, 10);
    const tierId = (rawTierId >= 1 && rawTierId <= 6) ? rawTierId : 1;
    const tier = PARITY_TIERS[tierId];
    const engagementType = opts.engagementType || "flagship";
    const rawServiceCount = parseInt(opts.serviceCount, 10);
    const serviceCount = (!isNaN(rawServiceCount) && rawServiceCount > 0) ? rawServiceCount : 1;

    let totalTrainingAmount = 0;
    let totalSupportAmount = 0;
    let isNegotiated = false;

    if (engagementType === "individual") {
      totalTrainingAmount = tier.indiv * serviceCount;
      totalSupportAmount = 0;
    } else if (engagementType === "custom") {
      isNegotiated = true;
      const rawCustom = typeof opts.customAmount === "number" ? opts.customAmount : parseFloat(opts.customAmount);
      totalTrainingAmount = (!isNaN(rawCustom) && rawCustom > 0) ? Math.round(rawCustom) : 0;
      totalSupportAmount = 0;
    } else {
      // Flagship or fallback default
      totalTrainingAmount = tier.flagship;
      totalSupportAmount = 0;
    }

    const trainingInstallmentAmount = Math.round((totalTrainingAmount * 0.10) * 100) / 100;
    const trainingMilestones = [];
    const dayStep = engagementType === "individual" ? 1 : 5;
    for (let i = 1; i <= 10; i++) {
      const dueDay = i * dayStep;
      trainingMilestones.push({
        milestoneIndex: i,
        percentage: 10,
        dueAfterDay: dueDay,
        amountUsd: trainingInstallmentAmount,
        dueTiming: "After Day " + dueDay + " completion and client verification",
        verifiedBeforePayment: true
      });
    }

    const supportMilestones = [];

    return {
      tierId: tier.id,
      tierName: tier.name,
      multiplier: tier.multiplier,
      discountPercent: tier.discount,
      engagementType: engagementType,
      serviceCount: serviceCount,
      totalTrainingAmountUsd: totalTrainingAmount,
      totalSupportAmountUsd: totalSupportAmount,
      totalEngagementAmountUsd: totalTrainingAmount + totalSupportAmount,
      upfrontDepositUsd: 0,
      isNegotiated: isNegotiated,
      trainingMilestones: trainingMilestones,
      supportMilestones: supportMilestones,
      trustTerms: TRUST_TERMS
    };
  }

  /**
   * Initializes the DOM controller for the Payment Schedule Widget.
   */
  function initDOMController() {
    if (typeof document === "undefined" || typeof document.getElementById !== "function") return;

    const widgetContainer = document.getElementById("payment-schedule-widget");
    if (!widgetContainer) return;

    const tierSelect = document.getElementById("sched-tier-select");
    const engSelect = document.getElementById("sched-eng-select");
    const customGroup = document.getElementById("sched-custom-group");
    const customInput = document.getElementById("sched-custom-input");
    const svcCountGroup = document.getElementById("sched-svc-count-group");
    const svcCountInput = document.getElementById("sched-svc-count-input");

    function renderSchedule() {
      const tierId = tierSelect ? parseInt(tierSelect.value, 10) : 1;
      const engType = engSelect ? engSelect.value : "flagship";
      const customAmount = customInput ? parseFloat(customInput.value) || 0 : 0;
      const serviceCount = svcCountInput ? parseInt(svcCountInput.value, 10) || 1 : 1;

      if (customGroup) {
        customGroup.style.display = engType === "custom" ? "block" : "none";
      }
      if (svcCountGroup) {
        svcCountGroup.style.display = engType === "individual" ? "block" : "none";
      }

      const schedule = calculateSchedule({
        tierId: tierId,
        engagementType: engType,
        customAmount: customAmount,
        serviceCount: serviceCount
      });

      const totalDisplay = document.getElementById("sched-total-display");
      const upfrontDisplay = document.getElementById("sched-upfront-display");
      const discountDisplay = document.getElementById("sched-discount-display");
      const licenseDisplay = document.getElementById("sched-license-display");
      const walkawayDisplay = document.getElementById("sched-walkaway-display");
      const negotiableNote = document.getElementById("sched-negotiable-note");
      const tableBody = document.getElementById("sched-milestones-body");

      if (totalDisplay) {
        totalDisplay.textContent = formatCurrency(schedule.totalEngagementAmountUsd);
      }
      if (upfrontDisplay) {
        upfrontDisplay.textContent = "$0 USD (Strictly Zero Upfront)";
      }
      if (discountDisplay) {
        if (schedule.isNegotiated) {
          discountDisplay.textContent = "Custom Negotiated Scope";
        } else if (schedule.discountPercent > 0) {
          discountDisplay.textContent = schedule.discountPercent + "% Parity Discount Applied";
        } else {
          discountDisplay.textContent = "Standard Commercial Rate";
        }
      }
      if (licenseDisplay) {
        licenseDisplay.textContent = "Apache 2.0 (Full Commercial Re-use Rights)";
      }
      if (walkawayDisplay) {
        walkawayDisplay.textContent = schedule.trustTerms.walkawayGuarantee;
      }
      if (negotiableNote) {
        negotiableNote.textContent = schedule.trustTerms.negotiability;
      }

      if (tableBody) {
        tableBody.innerHTML = "";
        schedule.trainingMilestones.forEach(function (m) {
          const tr = document.createElement("tr");
          tr.innerHTML = '<td style="padding:0.6rem 0.8rem;border-bottom:1px solid rgba(255,255,255,0.08);color:var(--cyan);font-weight:700;">Milestone ' + m.milestoneIndex + ' (10%)</td>' +
            '<td style="padding:0.6rem 0.8rem;border-bottom:1px solid rgba(255,255,255,0.08);color:var(--text-secondary);">' + m.dueTiming + '</td>' +
            '<td style="padding:0.6rem 0.8rem;border-bottom:1px solid rgba(255,255,255,0.08);color:#fff;font-weight:700;text-align:right;">' + formatCurrency(m.amountUsd) + '</td>';
          tableBody.appendChild(tr);
        });

        schedule.supportMilestones.forEach(function (sm) {
          const tr = document.createElement("tr");
          tr.innerHTML = '<td style="padding:0.6rem 0.8rem;border-bottom:1px solid rgba(255,255,255,0.08);color:var(--purple, #c084fc);font-weight:700;">Support Phase (Day ' + sm.dayDue + ')</td>' +
            '<td style="padding:0.6rem 0.8rem;border-bottom:1px solid rgba(255,255,255,0.08);color:var(--text-secondary);">' + sm.description + '</td>' +
            '<td style="padding:0.6rem 0.8rem;border-bottom:1px solid rgba(255,255,255,0.08);color:#fff;font-weight:700;text-align:right;">' + formatCurrency(sm.amountUsd) + '</td>';
          tableBody.appendChild(tr);
        });
      }
    }

    if (tierSelect) tierSelect.addEventListener("change", renderSchedule);
    if (engSelect) engSelect.addEventListener("change", renderSchedule);
    if (customInput) customInput.addEventListener("input", renderSchedule);
    if (svcCountInput) svcCountInput.addEventListener("input", renderSchedule);

    renderSchedule();
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", initDOMController);
    } else {
      initDOMController();
    }
  }

  return {
    calculateSchedule: calculateSchedule,
    formatCurrency: formatCurrency,
    initDOMController: initDOMController,
    PARITY_TIERS: PARITY_TIERS,
    TRUST_TERMS: TRUST_TERMS
  };
}));
