/**
 * pricing-calculator.js
 * Purchasing Power Parity (PPP) & Tier Rate Calculator Engine
 * Handles parity tier multipliers, client mode (individual vs business),
 * multi-service selection, extra seat calculations, and dynamic UI updates across pricing.html.
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    // Node.js environment
    module.exports = factory();
  } else {
    // Browser environment
    root.PricingCalculator = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {

  const PARITY_TIERS = {
    1: {
      id: 1,
      name: "Tier 1: Developed Nations (100% Standard Rate)",
      shortName: "Tier 1: Developed",
      multiplier: 1.0,
      discountPercent: 0,
      badgeText: "TIER 01 · HIGH-INCOME ECONOMIES",
      badgeColor: "#38bdf8",
      badgeBg: "rgba(56, 189, 248, 0.15)",
      badgeBorder: "rgba(56, 189, 248, 0.4)",
      note: "100% Standard Commercial Rate · Multiplier 1.0x",
      isCustom: false
    },
    2: {
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
    },
    3: {
      id: 3,
      name: "Tier 3: Lower-Middle & Low-Income Economies (50% Rate)",
      shortName: "Tier 3: Lower-Middle & Low-Income",
      multiplier: 0.50,
      discountPercent: 50,
      badgeText: "TIER 03 · LOWER-MIDDLE & LOW-INCOME",
      badgeColor: "#34d399",
      badgeBg: "rgba(52, 211, 153, 0.15)",
      badgeBorder: "rgba(52, 211, 153, 0.4)",
      note: "50% Parity Discount · 50% Base Rate · Multiplier 0.50x",
      isCustom: false
    },
    4: {
      id: 4,
      name: "Tier 4: Early-Stage Startups (50% Flat Discount)",
      shortName: "Tier 4: Startups",
      multiplier: 0.50,
      discountPercent: 50,
      badgeText: "TIER 04 · EARLY-STAGE STARTUPS",
      badgeColor: "#f472b6",
      badgeBg: "rgba(244, 114, 182, 0.15)",
      badgeBorder: "rgba(244, 114, 182, 0.4)",
      note: "50% Flat Startup Discount (<$1M Raised) · Multiplier 0.50x",
      isCustom: false
    },
    5: {
      id: 5,
      name: "Tier 5: Non-Profits & Academia (75% Discount / 25% Rate)",
      shortName: "Tier 5: Academia & Non-Profit",
      multiplier: 0.25,
      discountPercent: 75,
      badgeText: "TIER 05 · ACADEMIA & NON-PROFITS",
      badgeColor: "#c084fc",
      badgeBg: "rgba(192, 132, 252, 0.15)",
      badgeBorder: "rgba(192, 132, 252, 0.4)",
      note: "75% Flat Discount · 25% Base Rate · Multiplier 0.25x",
      isCustom: false
    },
    6: {
      id: 6,
      name: "Tier 6: Crisis-Affected & Under-Resourced Builders (Custom Subsidized Quote)",
      shortName: "Tier 6: Crisis-Affected & Under-Resourced",
      multiplier: 0.10,
      discountPercent: 90,
      badgeText: "TIER 06 · CRISIS-AFFECTED & UNDER-RESOURCED",
      badgeColor: "#00f0ff",
      badgeBg: "rgba(0, 240, 255, 0.18)",
      badgeBorder: "rgba(0, 240, 255, 0.45)",
      note: "Custom Subsidized Quotes & Micro-Rates (Up to 90%+ Relief)",
      isCustom: true
    }
  };

  const ALL_SERVICES_DATA = [
    { slug: "service-generative-ai-transformation", title: "Generative AI Transformation" },
    { slug: "service-local-llms-cost-slashing", title: "Local LLMs & Cost Slashing" },
    { slug: "service-ai-agents-orchestration", title: "Claude Agent Orchestration" },
    { slug: "service-agentic-ai-systems", title: "Hermes Agent Training" },
    { slug: "service-workflow-automations", title: "Workflow Automations" },
    { slug: "service-quantum-applications", title: "Quantum Applications (Experimental)" },
    { slug: "service-ai-budget-cost-limits-training", title: "AI Budget & Cost Control" },
    { slug: "service-enterprise-workforce-ai-training", title: "Enterprise Workforce AI Training" },
    { slug: "service-ai-developer-training", title: "AI Developer Training" },
    { slug: "service-advanced-ai-agents", title: "Advanced AI Agents" },
    { slug: "service-claude-basic-training", title: "Claude Basic Training" },
    { slug: "service-claude-advanced-training", title: "Claude Advanced Training" },
    { slug: "service-gemini-basic-training", title: "Google Gemini Basic Training" },
    { slug: "service-gemini-advanced-training", title: "Google Gemini Advanced Training" },
    { slug: "service-deep-research-notebooklm", title: "Deep Research & Gemini Notebook" },
    { slug: "service-hybrid-quantum-ai", title: "Hybrid Quantum-Classical AI" }
  ];

  /**
   * Pure calculation function
   * @param {Object} params
   * @param {number} params.tierId 1 to 6
   * @param {string} params.clientMode "business" or "individual"
   * @param {string} params.packageType "pick-services" | "pkg-business" | "pkg-personal" | "custom"
   * @param {Array<string>} params.selectedServices Array of service slugs
   * @param {number} params.participants Total participant count
   * @param {number} [params.customBase] Optional custom base price
   * @returns {Object} Calculated pricing details
   */
  function calculatePricing(params) {
    const tierId = params.tierId || 1;
    const tier = PARITY_TIERS[tierId] || PARITY_TIERS[1];
    const clientMode = params.clientMode || "business";
    const packageType = params.packageType || "pick-services";
    const selectedServices = params.selectedServices || ["service-generative-ai-transformation"];
    const ratePerSvc = clientMode === "business" ? 200 : 100;

    let basePrice = 2000;
    if (packageType === "pick-services") {
      const count = Math.max(1, selectedServices.length);
      basePrice = count * ratePerSvc;
    } else if (packageType === "pkg-business") {
      basePrice = 2000;
    } else if (packageType === "pkg-personal") {
      basePrice = 1000;
    } else if (packageType === "custom" && typeof params.customBase === 'number') {
      basePrice = params.customBase;
    }

    const matchingFlagshipPrice = clientMode === "business" ? 2000 : 1000;
    let flagshipCheaper = false;
    let flagshipPrice = matchingFlagshipPrice;

    if (packageType === "pick-services") {
      const count = Math.max(1, selectedServices.length);
      if (count * ratePerSvc >= matchingFlagshipPrice) {
        flagshipCheaper = true;
        flagshipPrice = matchingFlagshipPrice;
      }
    }

    let participants = clientMode === "individual" ? 1 : Math.max(1, parseInt(params.participants, 10) || 20);
    const extraParticipants = clientMode === "business" ? Math.max(0, participants - 20) : 0;
    const extraHeadRate = Math.round(extraParticipants * 100 * tier.multiplier);
    const baseTierPrice = Math.round(basePrice * tier.multiplier);
    const finalPrice = baseTierPrice + extraHeadRate;

    const baseSavings = Math.round(basePrice * (tier.discountPercent / 100));
    const extraSavings = Math.round(extraParticipants * 100 * (tier.discountPercent / 100));
    const totalSavings = baseSavings + extraSavings;

    return {
      tierId,
      tier,
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
      isCustom: tier.isCustom,
      flagshipCheaper,
      flagshipPrice
    };
  }

  /**
   * DOM Controller for pricing.html
   */
  function initDOMController() {
    if (typeof document === 'undefined') return;

    let activeTier = 1;
    let clientMode = "business"; // "business" or "individual"
    let standardBasePrice = 2000;
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
    const calcCountry = document.getElementById("calc-country");
    const calcTierButtons = document.querySelectorAll(".calc-tier-btn");
    const calcResultBox = document.getElementById("calc-result-box");
    const calcTierBadge = document.getElementById("calc-tier-badge");
    const calcDisplayPrice = document.getElementById("calc-display-price");
    const calcDisplayNote = document.getElementById("calc-display-note");
    const calcSavingsBadge = document.getElementById("calc-savings-badge");
    const calcBaseValue = document.getElementById("calc-base-value");
    const calcAppliedTier = document.getElementById("calc-applied-tier");
    const calcAdjustmentVal = document.getElementById("calc-adjustment-val");
    const calcCtaBtn = document.getElementById("calc-cta-btn");

    // Populate Checkbox Grid
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
          updateFromServicesSelection();
        });
      });
    }

    function updateFromServicesSelection() {
      const ratePerSvc = clientMode === "business" ? 200 : 100;
      const count = selectedServices.length;
      standardBasePrice = Math.max(count * ratePerSvc, ratePerSvc);

      if (selectedServiceCountLabel) {
        selectedServiceCountLabel.textContent = `${count} Service${count === 1 ? '' : 's'} Selected ($${(count * ratePerSvc).toLocaleString("en-US")} USD)`;
      }

      const tier = PARITY_TIERS[activeTier];
      if (calcCustomAmount && tier) {
        calcCustomAmount.value = Math.round(standardBasePrice * tier.multiplier);
      }

      updateCalculator();
    }

    if (btnSelectAllServices) {
      btnSelectAllServices.addEventListener("click", () => {
        selectedServices = ALL_SERVICES_DATA.map(s => s.slug);
        if (servicesCheckboxGrid) {
          servicesCheckboxGrid.querySelectorAll(".service-pick-cb").forEach(cb => cb.checked = true);
        }
        updateFromServicesSelection();
      });
    }

    if (btnClearServices) {
      btnClearServices.addEventListener("click", () => {
        selectedServices = [ALL_SERVICES_DATA[0].slug];
        if (servicesCheckboxGrid) {
          servicesCheckboxGrid.querySelectorAll(".service-pick-cb").forEach(cb => {
            cb.checked = (cb.value === selectedServices[0]);
          });
        }
        updateFromServicesSelection();
      });
    }

    if (btnModeBusiness && btnModeIndividual) {
      btnModeBusiness.addEventListener("click", () => setClientMode("business"));
      btnModeIndividual.addEventListener("click", () => setClientMode("individual"));
    }

    function setClientMode(mode) {
      clientMode = mode;
      if (mode === "business") {
        btnModeBusiness.classList.add("active");
        btnModeIndividual.classList.remove("active");
        if (calcParticipants) {
          calcParticipants.value = Math.max(20, parseInt(calcParticipants.value, 10) || 20);
          calcParticipants.disabled = false;
        }
        if (calcParticipantsHint) calcParticipantsHint.textContent = "Business: 1-20 Included (+$100/extra seat)";
        if (calcParticipantsDesc) calcParticipantsDesc.textContent = "Business base covers up to 20 participants (additional seats are $100 USD per seat scaled by country parity).";
      } else {
        btnModeIndividual.classList.add("active");
        btnModeBusiness.classList.remove("active");
        if (calcParticipants) {
          calcParticipants.value = 1;
          calcParticipants.disabled = true;
        }
        if (calcParticipantsHint) calcParticipantsHint.textContent = "Personal: Strictly 1 Person Only";
        if (calcParticipantsDesc) calcParticipantsDesc.textContent = "Personal / Individual engagements are strictly 1 person only (private 1-on-1 acceleration).";
      }

      if (calcPackage && calcPackage.value === "pick-services") {
        updateFromServicesSelection();
      } else if (calcPackage && calcPackage.value === "pkg-business" && mode === "individual") {
        calcPackage.value = "pkg-personal";
        standardBasePrice = 1000;
        const tier = PARITY_TIERS[activeTier];
        if (calcCustomAmount && tier) calcCustomAmount.value = Math.round(1000 * tier.multiplier);
        updateCalculator();
      } else if (calcPackage && calcPackage.value === "pkg-personal" && mode === "business") {
        calcPackage.value = "pkg-business";
        standardBasePrice = 2000;
        const tier = PARITY_TIERS[activeTier];
        if (calcCustomAmount && tier) calcCustomAmount.value = Math.round(2000 * tier.multiplier);
        updateCalculator();
      } else {
        updateCalculator();
      }
    }

    /**
     * Updates the 16 service cards in #individual-services-list
     * to reflect the active parity tier multiplier dynamically
     */
    function updateIndividualServiceCards(tier) {
      const indivCards = document.querySelectorAll("#individual-services-list .price-card-faq");
      if (!indivCards || indivCards.length === 0) return;

      const mult = tier.multiplier;

      indivCards.forEach(card => {
        const indivPriceEl = card.querySelector(".svc-indiv-price");
        const bizPriceEl = card.querySelector(".svc-biz-price");
        const indivTiersEl = card.querySelector(".svc-indiv-tiers");
        const bizTiersEl = card.querySelector(".svc-biz-tiers");

        if (indivPriceEl) {
          if (tier.isCustom) {
            indivPriceEl.textContent = "Custom quote - contact Thomas";
            indivPriceEl.style.color = "#00f0ff";
          } else {
            const calculated = Math.round(100 * mult);
            indivPriceEl.textContent = "$" + calculated.toLocaleString("en-US") + " USD" + (tier.id > 1 ? ` (T${tier.id})` : "");
            indivPriceEl.style.color = tier.badgeColor;
          }
        }

        if (bizPriceEl) {
          if (tier.isCustom) {
            bizPriceEl.textContent = "Custom quote - contact Thomas";
            bizPriceEl.style.color = "#00f0ff";
          } else {
            const calculated = Math.round(200 * mult);
            bizPriceEl.textContent = "$" + calculated.toLocaleString("en-US") + " USD" + (tier.id > 1 ? ` (T${tier.id})` : "");
            bizPriceEl.style.color = tier.badgeColor;
          }
        }

        // Highlight matching tier in tier preview string
        if (indivTiersEl) {
          indivTiersEl.style.color = tier.id > 1 ? "var(--text-secondary)" : "var(--text-muted)";
        }
        if (bizTiersEl) {
          bizTiersEl.style.color = tier.id > 1 ? "var(--text-secondary)" : "var(--text-muted)";
        }
      });
    }

    function updateCalculator() {
      if (!calcCustomAmount) return;

      const tier = PARITY_TIERS[activeTier];
      const result = calculatePricing({
        tierId: activeTier,
        clientMode: clientMode,
        packageType: calcPackage ? calcPackage.value : "pick-services",
        selectedServices: selectedServices,
        participants: calcParticipants ? calcParticipants.value : 20,
        customBase: standardBasePrice
      });

      // Update Base Value Row
      if (calcBaseValue) calcBaseValue.textContent = "$" + result.basePrice.toLocaleString("en-US") + " USD";
      if (calcAppliedTier) {
        calcAppliedTier.textContent = tier.shortName;
        calcAppliedTier.style.color = tier.badgeColor;
      }

      // Scope description
      if (calcScopeVal) {
        if (result.packageType === "pkg-business") {
          calcScopeVal.textContent = "Business Training Flagship (60 Days: 30d + 30d)";
          calcScopeVal.style.color = "var(--cyan)";
        } else if (result.packageType === "pkg-personal") {
          calcScopeVal.textContent = "Personal Training Flagship (30 Days: 15d + 15d)";
          calcScopeVal.style.color = "var(--gold)";
        } else if (result.packageType === "pick-services") {
          calcScopeVal.textContent = `${result.serviceCount} Selected Service${result.serviceCount === 1 ? '' : 's'} (${result.clientMode === "business" ? "Business" : "Individual"}, ${result.serviceCount * 5}d + ${result.serviceCount * 5}d support)`;
          calcScopeVal.style.color = result.clientMode === "business" ? "#38bdf8" : "var(--gold)";
        } else {
          calcScopeVal.textContent = "Custom Engineered Scope";
          calcScopeVal.style.color = "var(--text-secondary)";
        }
      }

      // Participants & Extra Head Scaling
      if (calcCohortVal) {
        if (result.clientMode === "individual") {
          calcCohortVal.textContent = "1 Person Only (Private 1-on-1)";
          calcCohortVal.style.color = "var(--gold)";
        } else {
          if (result.participants <= 20) {
            calcCohortVal.textContent = result.participants + (result.participants === 1 ? " Participant (Included in Base)" : " Participants (Included in Base)");
            calcCohortVal.style.color = "var(--gold)";
          } else {
            calcCohortVal.textContent = result.participants + " Heads (+" + result.extraParticipants + " Extra @ +$" + result.extraHeadRate.toLocaleString("en-US") + " USD)";
            calcCohortVal.style.color = "#38bdf8";
          }
        }
      }

      // Update Badge
      if (calcTierBadge) {
        calcTierBadge.textContent = tier.badgeText;
        calcTierBadge.style.color = tier.badgeColor;
        calcTierBadge.style.background = tier.badgeBg;
        calcTierBadge.style.borderColor = tier.badgeBorder;
      }

      // Dynamic Accent on Result Box Border
      if (calcResultBox) {
        calcResultBox.style.borderColor = tier.badgeColor;
        calcResultBox.style.boxShadow = "0 0 28px " + tier.badgeBg.replace("0.15", "0.25").replace("0.18", "0.25") + ", 0 10px 30px rgba(0,0,0,0.6)";
      }

      if (tier.isCustom) {
        // Tier 6 Subsidized
        if (calcDisplayPrice) {
          calcDisplayPrice.textContent = "Custom subsidized quote - contact Thomas";
          calcDisplayPrice.style.fontSize = "clamp(1.35rem, 2.4vw, 1.75rem)";
          calcDisplayPrice.style.color = tier.badgeColor;
          calcDisplayPrice.style.textShadow = "0 0 18px rgba(0, 240, 255, 0.45)";
        }

        if (calcDisplayNote) {
          calcDisplayNote.textContent = "Custom subsidized relief - contact Thomas directly";
          calcDisplayNote.style.color = "#7dd3fc";
        }

        if (calcAdjustmentVal) {
          calcAdjustmentVal.textContent = "Custom Subsidized Quote";
          calcAdjustmentVal.style.color = tier.badgeColor;
        }

        if (calcSavingsBadge) {
          calcSavingsBadge.innerHTML = '<span>Radical Equity: Custom subsidized rates tailored to your local budget</span>';
          calcSavingsBadge.style.color = tier.badgeColor;
          calcSavingsBadge.style.borderColor = tier.badgeBorder;
          calcSavingsBadge.style.background = tier.badgeBg;
        }

        if (calcCtaBtn) {
          calcCtaBtn.textContent = "Contact Thomas for Custom Subsidized Quote →";
          calcCtaBtn.style.borderColor = tier.badgeColor;
          calcCtaBtn.href = "contact.html?tier=" + encodeURIComponent(tier.name) + "&budget=Custom%20Subsidized%20Quote&participants=" + result.participants;
        }
      } else {
        // Tiers 1-5
        if (calcDisplayPrice) {
          calcDisplayPrice.textContent = "$" + result.finalPrice.toLocaleString("en-US") + " USD";
          calcDisplayPrice.style.fontSize = "clamp(2.1rem, 3.8vw, 2.85rem)";
          calcDisplayPrice.style.color = tier.badgeColor;
          calcDisplayPrice.style.textShadow = "0 0 22px " + tier.badgeBg.replace("0.15", "0.45");
        }

        if (calcDisplayNote) {
          if (result.extraParticipants > 0) {
            calcDisplayNote.innerHTML = tier.note + " · Plus " + result.extraParticipants + " extra seats (+$" + result.extraHeadRate.toLocaleString("en-US") + ")";
          } else {
            calcDisplayNote.innerHTML = tier.note + (result.clientMode === "business" ? " · Base covers up to 20 participants" : " · Strictly 1 person only");
          }
          calcDisplayNote.style.color = tier.badgeColor;
        }

        if (calcAdjustmentVal && calcSavingsBadge) {
          if (tier.discountPercent === 0) {
            calcAdjustmentVal.textContent = "$0 USD (Standard Rate)";
            calcAdjustmentVal.style.color = "var(--text-muted)";
            calcSavingsBadge.innerHTML = '<span>Standard Commercial Rate (0% Discount)</span>';
            calcSavingsBadge.style.color = "#38bdf8";
            calcSavingsBadge.style.borderColor = "rgba(56, 189, 248, 0.4)";
            calcSavingsBadge.style.background = "rgba(56, 189, 248, 0.12)";
          } else {
            calcAdjustmentVal.textContent = "-$" + result.totalSavings.toLocaleString("en-US") + " USD (" + tier.discountPercent + "% Off)";
            calcAdjustmentVal.style.color = "#34d399";
            calcSavingsBadge.innerHTML = '<span>You Save $' + result.totalSavings.toLocaleString("en-US") + ' USD (' + tier.discountPercent + '% Parity Subsidy)</span>';
            calcSavingsBadge.style.color = "#34d399";
            calcSavingsBadge.style.borderColor = "rgba(52, 211, 153, 0.4)";
            calcSavingsBadge.style.background = "rgba(52, 211, 153, 0.15)";
          }
        }

        if (calcCtaBtn) {
          calcCtaBtn.textContent = "Lock in " + tier.shortName + " Rate ($" + result.finalPrice.toLocaleString("en-US") + ") →";
          calcCtaBtn.style.borderColor = tier.badgeColor;
          calcCtaBtn.href = "contact.html?tier=" + encodeURIComponent(tier.name) + "&budget=" + result.finalPrice + "&participants=" + result.participants;
        }
      }

      // Flagship Cheaper Nudge (2.1)
      const calcFlagshipNudge = document.getElementById("calc-flagship-nudge");
      if (calcFlagshipNudge) {
        if (result.flagshipCheaper) {
          const isBiz = result.clientMode === "business";
          const pkgTitle = isBiz ? "Business Flagship" : "Personal Flagship";
          const flagshipCostStr = "$" + result.flagshipPrice.toLocaleString("en-US");
          const trainingDays = isBiz ? "30 days training + 30 days guidance" : "15 days training + 15 days guidance";
          const targetPkgVal = isBiz ? "pkg-business" : "pkg-personal";

          calcFlagshipNudge.style.display = "block";
          calcFlagshipNudge.innerHTML = `
            <div style="display:flex;flex-direction:column;gap:0.5rem;">
              <span style="font-weight:600;color:#fef08a;">The ${pkgTitle} (${flagshipCostStr}, ${trainingDays}) costs less than this selection. Switch to it?</span>
              <button type="button" id="btn-switch-flagship" style="align-self:flex-start;background:var(--gold);color:#020408;border:none;border-radius:6px;padding:0.4rem 0.85rem;font-weight:700;font-size:0.85rem;cursor:pointer;transition:transform 0.15s ease;">Switch to ${pkgTitle}</button>
            </div>
          `;
          const btnSwitch = document.getElementById("btn-switch-flagship");
          if (btnSwitch) {
            btnSwitch.onclick = function() {
              if (calcPackage) {
                calcPackage.value = targetPkgVal;
                calcPackage.dispatchEvent(new Event("change"));
              }
            };
          }
        } else {
          calcFlagshipNudge.style.display = "none";
          calcFlagshipNudge.innerHTML = "";
        }
      }

      // Update 16 individual service cards across the page
      updateIndividualServiceCards(tier);
    }

    function setActiveTier(tierNum) {
      activeTier = parseInt(tierNum, 10);
      calcTierButtons.forEach(btn => {
        if (parseInt(btn.getAttribute("data-tier"), 10) === activeTier) {
          btn.classList.add("active");
        } else {
          btn.classList.remove("active");
        }
      });

      if (calcCountry) {
        let matchingOpt = calcCountry.querySelector('option[data-tier="' + activeTier + '"]');
        if (matchingOpt && calcCountry.selectedOptions[0]?.getAttribute("data-tier") != activeTier) {
          matchingOpt.selected = true;
        }
      }

      const tier = PARITY_TIERS[activeTier];
      if (calcCustomAmount && tier) {
        const tierPrice = Math.round(standardBasePrice * tier.multiplier);
        calcCustomAmount.value = tierPrice;
      }

      updateCalculator();
    }

    if (calcPackage) {
      calcPackage.addEventListener("change", function () {
        const selectedOpt = this.options[this.selectedIndex];
        if (!selectedOpt) return;
        if (selectedOpt.value === "pick-services") {
          if (servicesMultiSelectWrap) servicesMultiSelectWrap.style.display = "block";
          updateFromServicesSelection();
        } else {
          if (servicesMultiSelectWrap) servicesMultiSelectWrap.style.display = "none";
          if (selectedOpt.value === "pkg-business") {
            setClientMode("business");
            standardBasePrice = 2000;
          } else if (selectedOpt.value === "pkg-personal") {
            setClientMode("individual");
            standardBasePrice = 1000;
          } else if (selectedOpt.value === "custom") {
            calcCustomAmount.focus();
            calcCustomAmount.select();
          }

          const tier = PARITY_TIERS[activeTier];
          if (calcCustomAmount && tier && selectedOpt.value !== "custom") {
            calcCustomAmount.value = Math.round(standardBasePrice * tier.multiplier);
          }
          updateCalculator();
        }
      });
    }

    if (calcCustomAmount) {
      calcCustomAmount.addEventListener("input", function () {
        const currentVal = parseFloat(this.value);
        const tier = PARITY_TIERS[activeTier];
        if (!isNaN(currentVal) && currentVal > 0 && tier) {
          standardBasePrice = Math.round(currentVal / (tier.multiplier || 1));
        }

        if (calcPackage) {
          let matched = false;
          for (let opt of calcPackage.options) {
            if (parseFloat(opt.getAttribute("data-price")) === standardBasePrice) {
              calcPackage.value = opt.value;
              matched = true;
              break;
            }
          }
          if (!matched) {
            calcPackage.value = "custom";
            if (servicesMultiSelectWrap) servicesMultiSelectWrap.style.display = "none";
          }
        }

        updateCalculator();
      });
    }

    if (calcParticipants) {
      calcParticipants.addEventListener("input", function () {
        updateCalculator();
      });
    }

    if (calcCountry) {
      calcCountry.addEventListener("change", function () {
        const selectedOption = this.options[this.selectedIndex];
        const tier = selectedOption.getAttribute("data-tier");
        if (tier) setActiveTier(tier);
      });
    }

    calcTierButtons.forEach(btn => {
      btn.addEventListener("click", function () {
        const tier = this.getAttribute("data-tier");
        setActiveTier(tier);
      });
    });

    window.selectTierAndScroll = function (tierNum) {
      setActiveTier(tierNum);
      const calcSection = document.getElementById("parity-calculator");
      if (calcSection) {
        calcSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }
      if (calcResultBox) {
        calcResultBox.classList.remove("tier-pulse");
        void calcResultBox.offsetWidth;
        calcResultBox.classList.add("tier-pulse");
      }
    };

    // Run initial update
    updateCalculator();
  }

  // Auto-init when DOM ready if in browser
  if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initDOMController);
    } else {
      initDOMController();
    }
  }

  return {
    PARITY_TIERS,
    ALL_SERVICES_DATA,
    calculatePricing,
    initDOMController
  };
}));
