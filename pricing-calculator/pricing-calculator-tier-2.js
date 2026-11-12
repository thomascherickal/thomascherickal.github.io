/**
 * pricing-calculator-tier-2.js
 * Tier 2 Pricing Calculator Engine: Tier 2: Upper-Middle
 * Dedicated Tier 1 Calculator with strict zero-emojis, zero-em-dash compliance.
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.PricingCalculatorTier2 = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {

  const TIER = {
    id: 2,
    name: "Tier 2: Upper-Middle-Income Economies (25% Discount / 75% Rate)",
    shortName: "Tier 2: Upper-Middle",
    multiplier: 0.75,
    discountPercent: 25,
    badgeText: "TIER 02 · UPPER-MIDDLE-INCOME",
    badgeColor: "#fbbf24",
    badgeBg: "rgba(251, 191, 36, 0.15)",
    badgeBorder: "rgba(251, 191, 36, 0.4)",
    note: "25% Parity Discount · 75% Base Rate · Multiplier 0.75x",
    isCustom: false
  };

  const ALL_SERVICES_DATA = [
    { slug: "service-generative-ai-transformation", title: "Generative AI Transformation" },
    { slug: "service-knowledge-graphs", title: "Knowledge Graphs" },
    { slug: "service-local-ai-for-sensitive-data", title: "Local AI for Sensitive Data" },
    { slug: "service-local-llms-cost-slashing", title: "Local LLMs & Cost Slashing" },
    { slug: "service-claude-agents-orchestration", title: "Claude Agent Orchestration" },
    { slug: "service-agentic-assistants", title: "Agentic Assistants Training" },
    { slug: "service-workflow-automations", title: "Workflow Automations" },
    { slug: "service-ai-budget-cost-control", title: "AI Budget & Cost Control" },
    { slug: "service-ai-developer-training", title: "AI Developer Training" },
    { slug: "service-advanced-ai-agents", title: "Advanced AI Agents" },
    { slug: "service-competitor-research", title: "Competitor Research & Intelligence" },
    { slug: "service-ai-assisted-leads-magnet", title: "AI Leads Magnet & Conversion Systems" },
    { slug: "service-ai-second-brain", title: "AI Second Brain" },
    { slug: "service-continuous-learning", title: "Continuous Learning & Synthesis" },
    { slug: "service-gemini-research-training", title: "Gemini Research Training" },
    { slug: "service-claude-research-training", title: "Claude Research Training" }
  ];

  function calculatePricing(params) {
    if (!params) params = {};
    const clientMode = params.clientMode === "individual" ? "individual" : "business";
    const packageType = params.packageType || "pick-services";
    const selectedServices = Array.isArray(params.selectedServices)
      ? params.selectedServices
      : (params.selectedServices ? [params.selectedServices] : ["service-generative-ai-transformation"]);
    const ratePerSvc = 300;

    let basePrice = 3000;
    if (packageType === "pick-services") {
      const count = selectedServices.length;
      basePrice = count * ratePerSvc;
    } else if (packageType === "pkg-business") {
      basePrice = 3000;
    } else if (packageType === "pkg-personal") {
      basePrice = 3000;
    } else if (packageType === "custom" && typeof params.customBase === 'number') {
      basePrice = Math.max(0, params.customBase);
    }

    const matchingFlagshipPrice = 3000;
    let flagshipCheaper = false;
    let flagshipPrice = matchingFlagshipPrice;

    if (packageType === "pick-services") {
      const count = selectedServices.length;
      if (count > 0 && count * ratePerSvc >= matchingFlagshipPrice) {
        flagshipCheaper = true;
        flagshipPrice = matchingFlagshipPrice;
      }
    }

    let rawParticipants = parseInt(params.participants, 10);
    if (isNaN(rawParticipants) || rawParticipants < 1) {
      rawParticipants = clientMode === "individual" ? 1 : 10;
    }
    let participants = clientMode === "individual" ? 1 : rawParticipants;
    const extraParticipants = (clientMode === "business" && basePrice > 0) ? Math.max(0, participants - 10) : 0;
    const extraHeadRate = Math.round(extraParticipants * 300 * TIER.multiplier);
    const baseTierPrice = Math.round(basePrice * TIER.multiplier);
    const finalPrice = basePrice === 0 ? 0 : baseTierPrice + extraHeadRate;

    const baseSavings = Math.round(basePrice * (TIER.discountPercent / 100));
    const extraSavings = Math.round(extraParticipants * 300 * (TIER.discountPercent / 100));
    const totalSavings = basePrice === 0 ? 0 : baseSavings + extraSavings;

    return {
      tierId: TIER.id,
      tier: TIER,
      clientMode,
      packageType,
      serviceCount: selectedServices.length,
      ratePerService: ratePerSvc,
      basePrice,
      participants,
      extraParticipants,
      extraHeadRate,
      baseTierPrice,
      finalPrice,
      totalSavings,
      isCustom: TIER.isCustom,
      flagshipCheaper,
      flagshipPrice
    };
  }

  function initDOMController() {
    if (typeof document === 'undefined') return;

    let clientMode = "business";
    let standardBasePrice = 3000;
    let selectedServices = ["service-generative-ai-transformation"];

    const btnModeBusiness = document.getElementById("btn-mode-business");
    const btnModeIndividual = document.getElementById("btn-mode-individual");
    const calcPackage = document.getElementById("calc-package");
    const servicesMultiSelectWrap = document.getElementById("services-multi-select-wrap");
    const servicesCheckboxGrid = document.getElementById("services-checkbox-grid");
    const selectedServiceCountLabel = document.getElementById("selected-service-count-label");
    const btnSelectAllServices = document.getElementById("btn-select-all-services");
    const btnClearServices = document.getElementById("btn-clear-services");
    const calcCustomAmount = document.getElementById("calc-custom-amount");
    const calcParticipants = document.getElementById("calc-participants");
    const calcParticipantsHint = document.getElementById("calc-participants-hint");
    const calcParticipantsDesc = document.getElementById("calc-participants-desc");
    const calcScopeVal = document.getElementById("calc-scope-val");
    const calcCohortVal = document.getElementById("calc-cohort-val");
    const calcResultBox = document.getElementById("calc-result-box");
    const calcTierBadge = document.getElementById("calc-tier-badge");
    const calcDisplayPrice = document.getElementById("calc-display-price");
    const calcDisplayNote = document.getElementById("calc-display-note");
    const calcSavingsBadge = document.getElementById("calc-savings-badge");
    const calcBaseValue = document.getElementById("calc-base-value");
    const calcAppliedTier = document.getElementById("calc-applied-tier");
    const calcAdjustmentVal = document.getElementById("calc-adjustment-val");
    const calcCtaBtn = document.getElementById("calc-cta-btn");
    const calcFlagshipNudge = document.getElementById("calc-flagship-nudge");

    if (servicesCheckboxGrid) {
      servicesCheckboxGrid.innerHTML = ALL_SERVICES_DATA.map((s) => {
        const isChecked = selectedServices.includes(s.slug) ? "checked" : "";
        return `
          <label style="display:flex;align-items:center;gap:0.45rem;background:rgba(255,255,255,0.02);padding:0.35rem 0.55rem;border-radius:6px;border:1px solid rgba(255,255,255,0.06);cursor:pointer;font-size:0.8rem;color:var(--text-secondary);">
            <input type="checkbox" class="service-pick-cb" value="${s.slug}" ${isChecked} style="accent-color:var(--cyan);cursor:pointer;" />
            <span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${s.title}</span>
          </label>
        `;
      }).join("");

      servicesCheckboxGrid.querySelectorAll(".service-pick-cb").forEach(cb => {
        cb.addEventListener("change", function () {
          if (this.checked) {
            if (!selectedServices.includes(this.value)) selectedServices.push(this.value);
          } else {
            selectedServices = selectedServices.filter(slug => slug !== this.value);
          }
          updateAll();
        });
      });
    }

    if (btnSelectAllServices) {
      btnSelectAllServices.addEventListener("click", () => {
        selectedServices = ALL_SERVICES_DATA.map(s => s.slug);
        if (servicesCheckboxGrid) {
          servicesCheckboxGrid.querySelectorAll(".service-pick-cb").forEach(cb => cb.checked = true);
        }
        updateAll();
      });
    }

    if (btnClearServices) {
      btnClearServices.addEventListener("click", () => {
        selectedServices = [];
        if (servicesCheckboxGrid) {
          servicesCheckboxGrid.querySelectorAll(".service-pick-cb").forEach(cb => cb.checked = false);
        }
        updateAll();
      });
    }

    function setClientMode(newMode) {
      clientMode = newMode;
      if (clientMode === "individual") {
        if (btnModeIndividual) btnModeIndividual.classList.add("active");
        if (btnModeBusiness) btnModeBusiness.classList.remove("active");
        if (calcParticipants) {
          calcParticipants.value = 1;
          calcParticipants.disabled = true;
          calcParticipants.style.opacity = "0.5";
        }
        if (calcParticipantsHint) calcParticipantsHint.textContent = "Personal: Strictly 1 Person Only";
        if (calcParticipantsDesc) calcParticipantsDesc.textContent = "Individual/Personal tracks are strictly 1 person only (private 1-on-1 acceleration).";
        if (calcPackage && calcPackage.value === "pkg-business") {
          calcPackage.value = "pkg-personal";
          standardBasePrice = 3000;
        }
      } else {
        if (btnModeBusiness) btnModeBusiness.classList.add("active");
        if (btnModeIndividual) btnModeIndividual.classList.remove("active");
        if (calcParticipants) {
          calcParticipants.disabled = false;
          calcParticipants.style.opacity = "1";
          if (parseInt(calcParticipants.value, 10) <= 1) calcParticipants.value = 10;
        }
        if (calcParticipantsHint) calcParticipantsHint.textContent = "Business: 1-10 Included (+$300/extra seat)";
        if (calcParticipantsDesc) calcParticipantsDesc.textContent = "Business base covers up to 10 participants (additional seats are $300 USD per seat scaled by country parity).";
        if (calcPackage && calcPackage.value === "pkg-personal") {
          calcPackage.value = "pkg-business";
          standardBasePrice = 3000;
        }
      }
      updateAll();
    }

    if (btnModeBusiness) btnModeBusiness.addEventListener("click", () => setClientMode("business"));
    if (btnModeIndividual) btnModeIndividual.addEventListener("click", () => setClientMode("individual"));

    if (calcPackage) {
      calcPackage.addEventListener("change", function () {
        const val = this.value;
        const selectedOpt = this.options[this.selectedIndex];
        if (val === "pick-services") {
          if (servicesMultiSelectWrap) servicesMultiSelectWrap.style.display = "block";
          if (calcCustomAmount) calcCustomAmount.closest(".calc-field-group").style.display = "none";
        } else if (val === "custom") {
          if (servicesMultiSelectWrap) servicesMultiSelectWrap.style.display = "none";
          if (calcCustomAmount) {
            calcCustomAmount.closest(".calc-field-group").style.display = "block";
            standardBasePrice = parseFloat(calcCustomAmount.value) || 3000;
          }
        } else {
          if (servicesMultiSelectWrap) servicesMultiSelectWrap.style.display = "none";
          if (calcCustomAmount) calcCustomAmount.closest(".calc-field-group").style.display = "none";
          standardBasePrice = parseFloat(selectedOpt.getAttribute("data-price")) || 3000;
          if (val === "pkg-business") {
            setClientMode("business");
            standardBasePrice = 3000;
          } else if (val === "pkg-personal") {
            setClientMode("individual");
            standardBasePrice = 3000;
          }
        }
        updateAll();
      });
    }

    if (calcCustomAmount) {
      calcCustomAmount.addEventListener("input", function () {
        standardBasePrice = Math.max(0, parseFloat(this.value) || 0);
        updateAll();
      });
    }

    if (calcParticipants) {
      calcParticipants.addEventListener("input", function () {
        updateAll();
      });
    }

    function updateAll() {
      const packageType = calcPackage ? calcPackage.value : "pkg-business";
      let parts = 10;
      if (clientMode === "individual") {
        parts = 1;
      } else if (calcParticipants) {
        parts = parseInt(calcParticipants.value, 10) || 10;
      }

      const calcParams = {
        tierId: 2,
        clientMode,
        packageType,
        selectedServices,
        participants: parts,
        customBase: standardBasePrice
      };

      const result = calculatePricing(calcParams);

      if (selectedServiceCountLabel) {
        const c = selectedServices.length;
        selectedServiceCountLabel.textContent = `${c} Service${c === 1 ? '' : 's'} Selected ($${result.basePrice.toLocaleString()} USD)`;
      }

      if (calcResultBox) {
        calcResultBox.style.borderColor = TIER.badgeColor;
        calcResultBox.style.boxShadow = "0 0 28px " + TIER.badgeBg.replace("0.15", "0.25") + ", 0 10px 30px rgba(0,0,0,0.6)";
      }

      if (calcTierBadge) {
        calcTierBadge.textContent = TIER.badgeText;
        calcTierBadge.style.color = TIER.badgeColor;
        calcTierBadge.style.background = TIER.badgeBg;
        calcTierBadge.style.borderColor = TIER.badgeBorder;
      }

      if (calcDisplayPrice) {
        calcDisplayPrice.textContent = `$${result.finalPrice.toLocaleString()} USD`;
      }

      if (calcDisplayNote) {
        calcDisplayNote.textContent = TIER.note;
        calcDisplayNote.style.color = "#7dd3fc";
      }

      if (calcSavingsBadge) {
        calcSavingsBadge.innerHTML = `<span>Saving 25% with Country Parity (-${result.totalSavings.toLocaleString()} USD)</span>`;
        calcSavingsBadge.style.background = "rgba(56, 189, 248, 0.12)";
        calcSavingsBadge.style.borderColor = "rgba(56, 189, 248, 0.35)";
      }

      if (calcFlagshipNudge) {
        if (result.flagshipCheaper) {
          calcFlagshipNudge.style.display = "block";
          calcFlagshipNudge.innerHTML = `<strong>Flagship Package Recommendation:</strong> Selecting ${result.serviceCount} services equals $${(result.serviceCount * result.ratePerService).toLocaleString()} USD. Our comprehensive Flagship Training covers the full engineering stack at $${result.flagshipPrice.toLocaleString()} USD.`;
        } else {
          calcFlagshipNudge.style.display = "none";
        }
      }

      if (calcBaseValue) {
        calcBaseValue.textContent = `$${result.basePrice.toLocaleString()} USD`;
      }

      if (calcScopeVal) {
        if (packageType === "pick-services") {
          calcScopeVal.textContent = `${result.serviceCount} Services Picked (5d + 5d each)`;
        } else if (packageType === "pkg-business") {
          calcScopeVal.textContent = "Business Training Flagship (60 Days: 30d + 30d, 4h/day)";
        } else if (packageType === "pkg-personal") {
          calcScopeVal.textContent = "Personal Training Flagship (60 Days: 30d + 30d, 4h/day)";
        } else {
          calcScopeVal.textContent = "Custom Scope Package";
        }
      }

      if (calcCohortVal) {
        if (clientMode === "individual") {
          calcCohortVal.textContent = "1 Participant (1-on-1 Dedicated)";
        } else if (result.extraParticipants > 0) {
          calcCohortVal.textContent = `${result.participants} Seats (10 Included + ${result.extraParticipants} Extra @ $${result.extraHeadRate.toLocaleString()} USD)`;
        } else {
          calcCohortVal.textContent = `${result.participants} Seats (Included in Base Rate)`;
        }
      }

      if (calcAppliedTier) {
        calcAppliedTier.textContent = TIER.shortName;
        calcAppliedTier.style.color = TIER.badgeColor;
      }

      if (calcAdjustmentVal) {
        calcAdjustmentVal.textContent = "-25% (-${result.totalSavings.toLocaleString()} USD)";
        calcAdjustmentVal.style.color = "#34d399";
      }

      if (calcCtaBtn) {
        calcCtaBtn.textContent = `Lock in Tier 2 Parity Rate ($${result.finalPrice.toLocaleString()}) →`;
        calcCtaBtn.href = `../contact.html?tier=2&budget=${result.finalPrice}`;
      }
    }

    updateAll();
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", initDOMController);
    } else {
      initDOMController();
    }
  }

  return {
    tier: TIER,
    calculatePricing,
    initDOMController,
    ALL_SERVICES_DATA
  };
}));
