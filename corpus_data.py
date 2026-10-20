"""
corpus_data.py
--------------
Realistic dataset of 60+ multi-sentence news/incident snippets across 4 categories:
  1. AI & Cybersecurity
  2. Finance & Markets
  3. Enterprise Cloud & Hardware
  4. Policy & Regulation
"""

CATEGORIES = [
    "AI & Cybersecurity",
    "Finance & Markets",
    "Enterprise Cloud & Hardware",
    "Policy & Regulation",
]

CORPUS = [
    # ─────────────────────────────────────────────────────────────
    # Category 0 : AI & Cybersecurity
    # ─────────────────────────────────────────────────────────────
    {
        "id": 0,
        "title": "APT Group Deploys AI-Powered Malware",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "A sophisticated advanced persistent threat group has deployed a novel malware strain "
            "that leverages on-device large language models to craft highly convincing spear-phishing "
            "emails. Security researchers at CrowdStrike discovered the campaign targeting Fortune 500 "
            "financial institutions. The malware autonomously rewrites its payload signatures after each "
            "execution cycle, evading signature-based antivirus detection with alarming efficiency. "
            "Incident responders noted that the adversary used RAG pipelines to ingest victim LinkedIn "
            "profiles before generating personalised lures."
        ),
    },
    {
        "id": 1,
        "title": "Generative AI Backdoors in Open-Source Models",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "Researchers at MIT CSAIL revealed that seventeen popular Hugging Face model checkpoints "
            "contain hidden backdoors activated by specific trigger phrases. When deployed in enterprise "
            "chatbots the backdoored models exfiltrate conversation logs to remote command-and-control "
            "servers. The vulnerability affects transformer architectures fine-tuned on curated instruction "
            "datasets. Hugging Face has since introduced mandatory model-scanning pipelines using "
            "differential privacy audits and attribution tracing to detect anomalous gradient patterns."
        ),
    },
    {
        "id": 2,
        "title": "Zero-Day Exploit Targets Neural Inference Chips",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "A critical zero-day vulnerability in the firmware of widely-used AI inference accelerators "
            "allows attackers to inject malicious microcode during model loading. The flaw, tracked as "
            "CVE-2026-4412, affects chips manufactured by three major silicon vendors. Threat actors can "
            "exploit the bug remotely via a crafted ONNX model file sent over an exposed model-serving "
            "API endpoint. CISA issued an emergency directive ordering federal agencies to patch all "
            "affected inference servers within 72 hours of disclosure."
        ),
    },
    {
        "id": 3,
        "title": "Autonomous Red-Team AI Breaks Enterprise Perimeter",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "A penetration-testing firm demonstrated that their autonomous red-team AI completed a full "
            "kill-chain attack against a simulated enterprise network in under four minutes. The agent "
            "used reinforcement learning to adapt its lateral-movement strategy after each defensive "
            "response, exploiting misconfigurations in Active Directory and Kubernetes RBAC policies. "
            "The demonstration raised urgent questions about the readiness of blue teams to defend "
            "against machine-speed adversaries. NIST subsequently published a draft framework for "
            "AI-accelerated threat modelling."
        ),
    },
    {
        "id": 4,
        "title": "LLM Prompt Injection Targets Banking Chatbots",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "Security analysts discovered a widespread prompt-injection campaign targeting LLM-powered "
            "customer service chatbots at eight major retail banks. Attackers embedded hidden instructions "
            "inside uploaded PDF documents, causing chatbots to reveal internal system prompts and bypass "
            "fraud-detection guardrails. Two banks reported unauthorized fund transfers totalling over "
            "six million dollars before the vulnerability was identified. Vendors are now evaluating "
            "input-sanitisation layers and instruction-hierarchy enforcement mechanisms."
        ),
    },
    {
        "id": 5,
        "title": "AI-Driven Deepfake Fraud Hits Corporate Executives",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "Criminals used real-time audio deepfake technology to impersonate the CEO of a British "
            "multinational during a video conference call, successfully instructing the CFO to wire "
            "25 million dollars to overseas accounts. Forensic analysts confirmed the synthetic voice "
            "achieved a 94 percent similarity score against the executive's authentic recordings. "
            "The incident has accelerated enterprise adoption of voice-biometric liveness detection "
            "and out-of-band transaction verification protocols across the financial sector."
        ),
    },
    {
        "id": 6,
        "title": "Ransomware Group Uses AI for Target Profiling",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "A prolific ransomware syndicate has integrated machine-learning models into their initial "
            "access broker pipeline to automatically score victim organizations by estimated ransom "
            "payment probability. The models analyse public financial filings, employee count data, "
            "and cloud asset exposure to prioritise high-value targets. Law enforcement agencies in "
            "five countries coordinated Operation DataShield, resulting in the arrest of twelve "
            "individuals and the seizure of 47 cryptocurrency wallets worth approximately 83 million "
            "dollars."
        ),
    },
    {
        "id": 7,
        "title": "Federated Learning System Poisoned in Healthcare Network",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "A federated learning consortium connecting 34 hospitals discovered that a compromised "
            "participating node had been injecting adversarial gradient updates for six weeks, "
            "systematically degrading diagnostic accuracy for rare cardiovascular conditions. "
            "The attack exploited weak authentication in the aggregation server's gRPC interface. "
            "The incident highlights the critical need for Byzantine-fault-tolerant aggregation "
            "algorithms and differential privacy budgets in sensitive medical AI deployments."
        ),
    },
    {
        "id": 8,
        "title": "Supply-Chain Attack Embeds Keylogger in AI SDK",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "Threat actors compromised the PyPI account of a popular AI development SDK maintainer, "
            "publishing a malicious version that silently installed a keylogger alongside the legitimate "
            "package. Over 120,000 developer machines downloaded the trojanized release within 48 hours "
            "before the package was removed. Stolen credentials were subsequently used to pivot into "
            "corporate development environments. The incident renewed calls for mandatory cryptographic "
            "signing of all package releases on major software registries."
        ),
    },
    {
        "id": 9,
        "title": "Neural Network Watermarking Thwarts Model Theft",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "A team at Stanford unveiled a robust neural-network watermarking scheme that embeds "
            "invisible ownership signatures into model weights without measurable performance degradation. "
            "The technique survived model distillation, pruning, and fine-tuning attacks in 98 percent "
            "of test scenarios. Several AI companies have begun licensing the technology to protect "
            "proprietary models from extraction attacks conducted through public inference APIs, "
            "a growing threat as capable open-source alternatives proliferate."
        ),
    },
    {
        "id": 10,
        "title": "AI Traffic Analysis Detects Encrypted C2 Channels",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "A next-generation network detection platform powered by transformer-based traffic classifiers "
            "successfully identified covert command-and-control communications hidden within HTTPS streams "
            "to legitimate cloud providers. The model was trained on 18 months of enterprise network flow "
            "data and achieved a 99.3 percent true-positive rate with sub-millisecond inference latency. "
            "Early adopters in the defence and energy sectors reported a 40 percent reduction in median "
            "dwell time following deployment."
        ),
    },
    {
        "id": 11,
        "title": "Multimodal AI Weaponized for Social Engineering Campaigns",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "State-sponsored actors deployed a multimodal AI system capable of generating synthetic "
            "images, fabricated audio clips, and contextually accurate text in a coordinated social "
            "engineering campaign targeting defence contractors. The platform automatically tailored "
            "each phishing artefact to the recipient's role and recent public statements. Attribution "
            "analysis linked the campaign infrastructure to previously documented nation-state groups "
            "known for espionage operations against critical infrastructure sectors."
        ),
    },
    {
        "id": 12,
        "title": "AI-Assisted Vulnerability Discovery Accelerates Patch Cycles",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "Google Project Zero integrated a code-analysis LLM into its vulnerability research pipeline, "
            "enabling analysts to discover and triage 400 percent more security bugs per quarter compared "
            "to manual review workflows. The system automatically generates proof-of-concept exploits "
            "for discovered vulnerabilities, enabling vendors to validate patches faster. The team "
            "published full methodology details, sparking debate about responsible disclosure timelines "
            "when AI can dramatically accelerate exploit development."
        ),
    },
    {
        "id": 13,
        "title": "Quantum-Resistant AI Encryption Deployed at Scale",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "A major cloud provider became the first to deploy post-quantum cryptographic algorithms "
            "natively within its AI platform's inference API, protecting model weights and inference "
            "traffic against harvest-now-decrypt-later attacks. The migration to NIST-standardised "
            "lattice-based encryption schemes affected over two million customer workloads. Security "
            "architects praised the seamless rollout but noted that edge inference devices with "
            "constrained compute budgets remain a significant challenge for post-quantum adoption."
        ),
    },
    {
        "id": 14,
        "title": "Insider Threat Uses AI Model Access to Exfiltrate IP",
        "category": "AI & Cybersecurity",
        "label": 0,
        "text": (
            "A disgruntled machine learning engineer at a semiconductor startup used their privileged "
            "model-training access to embed proprietary chip design blueprints as steganographic "
            "payloads within trained model weights before resigning. The exfiltration evaded data-loss "
            "prevention tools because the transmission occurred as a routine model checkpoint upload. "
            "Forensic investigators recovered the data using reverse-engineered steganography analysis, "
            "and the case is now being prosecuted under trade-secret theft statutes."
        ),
    },

    # ─────────────────────────────────────────────────────────────
    # Category 1 : Finance & Markets
    # ─────────────────────────────────────────────────────────────
    {
        "id": 15,
        "title": "Federal Reserve Signals Gradual Rate Normalisation",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "Federal Reserve Chair announced a cautious pivot toward interest-rate normalisation after "
            "inflation metrics sustained their downward trajectory for five consecutive quarters. "
            "Ten-year Treasury yields fell twelve basis points following the statement as bond markets "
            "priced in an 82 percent probability of a 25-basis-point cut at the next FOMC meeting. "
            "Equity indices rallied broadly, with the S&P 500 closing 1.7 percent higher and the "
            "Nasdaq Composite gaining 2.4 percent on renewed growth-stock enthusiasm."
        ),
    },
    {
        "id": 16,
        "title": "Algorithmic Trading Flash Crash Erases 800 Billion Dollars",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "A cascading failure among high-frequency trading algorithms triggered a flash crash that "
            "briefly erased 800 billion dollars of market capitalisation across US equities in under "
            "fourteen minutes. The event was precipitated by a misconfigured momentum indicator in a "
            "major quant fund's execution engine amplifying a routine sell-off. Circuit breakers halted "
            "trading on twelve major exchanges. Regulators are now reviewing the adequacy of existing "
            "market-stability mechanisms in the face of increasingly autonomous trading systems."
        ),
    },
    {
        "id": 17,
        "title": "Emerging Market Debt Crisis Spreads to Three Economies",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "Soaring dollar-denominated debt servicing costs have pushed three emerging market economies "
            "to the brink of sovereign default as the US dollar continues its prolonged appreciation "
            "cycle. IMF emergency credit facilities worth 42 billion dollars have been mobilised, "
            "conditional on structural adjustment programmes. Contagion fears sent credit default swap "
            "spreads surging across the frontier-market bond complex. Economists warn that tight global "
            "financial conditions may prolong recovery timelines for already-fragile developing nations."
        ),
    },
    {
        "id": 18,
        "title": "Central Bank Digital Currency Pilot Launches in G7",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "The Bank of England initiated a large-scale retail CBDC pilot involving 200,000 volunteer "
            "participants drawn from demographically diverse regions. The digital pound prototype runs "
            "on a permissioned distributed ledger and supports programmable payment logic for welfare "
            "disbursements and carbon-credit tokenisation. Privacy advocates raised concerns about "
            "transaction surveillance capabilities. The six-month trial will inform the government's "
            "decision on whether to proceed with full-scale digital currency issuance."
        ),
    },
    {
        "id": 19,
        "title": "Hedge Fund Leverages NLP to Predict Earnings Surprises",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "A quantitative hedge fund disclosed that its natural language processing system, trained on "
            "eight years of earnings call transcripts, SEC filings, and management commentary, achieved "
            "a 67 percent accuracy rate in predicting earnings-per-share surprises. The alpha-generating "
            "strategy outperformed the benchmark by 340 basis points annualised over a three-year "
            "backtest. Linguistic sentiment indicators derived from CFO language patterns proved "
            "especially predictive during volatile macro environments."
        ),
    },
    {
        "id": 20,
        "title": "Cryptocurrency Exchange Files for Bankruptcy Amid Fraud Allegations",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "A major cryptocurrency exchange filed for Chapter 11 bankruptcy protection after regulators "
            "alleged that customer funds had been systematically commingled with the firm's proprietary "
            "trading operations. Creditors representing over nine billion dollars in claims have formed "
            "an ad hoc committee to maximise recovery. The collapse sent Bitcoin plunging 18 percent "
            "and triggered liquidations across leveraged altcoin positions. Congressional hearings are "
            "scheduled to examine gaps in digital-asset custody and segregation requirements."
        ),
    },
    {
        "id": 21,
        "title": "ESG Investment Funds Underperform During Energy Supercycle",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "Environmental, social and governance-focused equity funds lagged their conventional "
            "benchmark counterparts by an average of 620 basis points during the recent energy "
            "supercycle as fossil-fuel stocks surged on supply constraints. Institutional investors "
            "are reassessing portfolio construction methodologies and benchmark exclusion lists. "
            "Several prominent ESG fund managers reduced their underweight positions in traditional "
            "energy companies while maintaining commitments to long-term decarbonisation objectives "
            "and engagement-based stewardship strategies."
        ),
    },
    {
        "id": 22,
        "title": "Private Equity Dry Powder Reaches Record 3.9 Trillion Dollars",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "Global private equity dry powder reached a record 3.9 trillion dollars as deal-making "
            "remained depressed by elevated interest rates and compressed valuation multiples. "
            "Buyout funds have increasingly deployed capital into continuation vehicles and secondaries "
            "markets as traditional exit routes via IPOs and strategic sales remained constrained. "
            "Limited partners are intensifying scrutiny of fund performance metrics and are demanding "
            "more frequent liquidity mechanisms. Analysts expect activity to accelerate once monetary "
            "policy easing cycles broaden globally."
        ),
    },
    {
        "id": 23,
        "title": "AI-Optimised Credit Scoring Expands Financial Inclusion",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "A fintech lender deployed a machine-learning credit-scoring model incorporating 3,200 "
            "alternative data features including utility payments, rental history, and mobile usage "
            "patterns to assess creditworthiness for the 1.4 billion unbanked consumers globally. "
            "Default rates on loans issued under the new model were 28 percent lower than traditional "
            "FICO-score-based underwriting for comparable income brackets. Regulators issued guidance "
            "requiring explainability reports for all adverse-action decisions generated by AI credit "
            "models."
        ),
    },
    {
        "id": 24,
        "title": "Commodity Supercycle Drives Metals ETF Inflows",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "Lithium, cobalt, and rare earth element ETFs attracted record inflows of 14.2 billion "
            "dollars in the first quarter as investors positioned for sustained demand from electric "
            "vehicle battery manufacturing and grid-scale energy storage buildout. Copper futures "
            "breached the five-dollar-per-pound threshold for the first time in seven years on "
            "constrained mine supply. Analysts at Goldman Sachs revised their 12-month price targets "
            "for base metals upward by an average of 23 percent."
        ),
    },
    {
        "id": 25,
        "title": "Stablecoin Regulation Bill Advances Through US Senate",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "The Stablecoin Transparency and Reliability Act advanced through the Senate Banking "
            "Committee with bipartisan support, mandating one-to-one reserve backing and monthly "
            "third-party attestation for all dollar-pegged stablecoins with market capitalisations "
            "exceeding one billion dollars. Issuers failing compliance deadlines face civil penalties "
            "of up to 100,000 dollars per day. Industry groups broadly welcomed the regulatory clarity "
            "while opposing provisions that would prohibit algorithmic stablecoins entirely."
        ),
    },
    {
        "id": 26,
        "title": "Sovereign Wealth Fund Pivots to AI Infrastructure Assets",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "Norway's Government Pension Fund Global announced a strategic 5-percent allocation to "
            "AI infrastructure assets encompassing data centres, GPU clusters, and subsea fibre-optic "
            "cable networks. The reallocation, worth approximately 95 billion dollars, reflects "
            "conviction that AI compute capacity represents a generational infrastructure investment "
            "opportunity comparable to the early internet build-out. Energy efficiency benchmarks and "
            "carbon-neutrality commitments will govern which assets qualify for inclusion."
        ),
    },
    {
        "id": 27,
        "title": "Options Market Signals Volatility Spike Ahead of Earnings",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "Implied volatility in short-dated options contracts for semiconductor and hyperscaler "
            "stocks surged to 18-month highs as investors hedged concentrated positions ahead of the "
            "busiest week of the earnings season. Put-to-call ratios on major AI chipmakers rose "
            "sharply, suggesting institutional caution despite recent all-time highs in underlying "
            "equity prices. Derivatives strategists noted that the term structure inversion indicated "
            "near-term hedging demand rather than sustained bearish conviction."
        ),
    },
    {
        "id": 28,
        "title": "Retail Investor Surge Reshapes Equity Market Microstructure",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "Commission-free brokerage platforms have driven retail investor participation to record "
            "highs, with individual traders now accounting for 28 percent of US equity market volume. "
            "Academic research confirms that coordinated retail trading via social media platforms "
            "exhibits persistent price-impact effects lasting up to three trading sessions. Market "
            "makers have adapted their inventory management algorithms to account for herding behaviours "
            "observed in retail order flow, widening spreads selectively during high-sentiment episodes."
        ),
    },
    {
        "id": 29,
        "title": "Inflation Swaps Pricing Long-Term Structural Shift",
        "category": "Finance & Markets",
        "label": 1,
        "text": (
            "Breakeven inflation rates embedded in five-year five-year forward inflation swaps have "
            "drifted above 2.8 percent, suggesting bond markets anticipate structurally higher price "
            "levels driven by deglobalisation trends, energy transition costs, and persistent labour "
            "market tightness. Central bank credibility in anchoring long-run inflation expectations "
            "is being tested as wage-price spiral risks re-emerge in service-dominated economies. "
            "TIPS demand has surged as institutional portfolios reassess real-return requirements."
        ),
    },

    # ─────────────────────────────────────────────────────────────
    # Category 2 : Enterprise Cloud & Hardware
    # ─────────────────────────────────────────────────────────────
    {
        "id": 30,
        "title": "Hyperscaler Data Centre Outage Disrupts Global SaaS",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "A nine-hour outage at a major hyperscaler's eastern US data centre region cascaded to "
            "disrupt over 14,000 dependent SaaS platforms worldwide after a cooling-system failure "
            "forced an emergency shutdown of three availability zones. Estimated economic losses "
            "exceeded 600 million dollars in lost productivity and transactional revenue. Post-incident "
            "reports revealed that the cooling failure was triggered by a faulty firmware update applied "
            "to automated thermal-management controllers. Customers affected by the outage invoked "
            "service-level agreement credits totalling over 80 million dollars."
        ),
    },
    {
        "id": 31,
        "title": "Next-Generation GPU Architecture Doubles AI Throughput",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "A leading semiconductor company unveiled its next-generation data centre GPU architecture, "
            "delivering 2.1 times the transformer model training throughput compared to its predecessor "
            "at equivalent thermal design power. The chip integrates high-bandwidth memory with 3D "
            "stacking technology and a dedicated sparsity-acceleration engine that automatically "
            "exploits zero-weight patterns in large neural networks. Cloud providers began accepting "
            "pre-orders with estimated delivery timelines of nine months, reflecting continued "
            "constrained manufacturing capacity at advanced process nodes."
        ),
    },
    {
        "id": 32,
        "title": "Kubernetes Security Misconfiguration Exposes Fortune 500 Data",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "Security researchers discovered that over 3,800 Kubernetes clusters belonging to Fortune "
            "500 enterprises were reachable via the public internet with default or weak authentication "
            "configurations, exposing sensitive workload metadata and container images. The scan was "
            "conducted using passive network reconnaissance tools and revealed persistent misconfiguration "
            "patterns stemming from developer self-service cluster provisioning without mandatory security "
            "baselines. Cloud-native security platforms offering continuous compliance posture monitoring "
            "saw a surge in enterprise procurement following the disclosure."
        ),
    },
    {
        "id": 33,
        "title": "Sovereign Cloud Mandates Reshape Global Multi-Cloud Strategy",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "New data sovereignty regulations in the European Union, India, and Brazil are forcing "
            "multinational enterprises to redesign cloud architectures to ensure that regulated data "
            "never traverses foreign jurisdiction boundaries. Hyperscalers are accelerating deployment "
            "of locally operated sovereign cloud regions with dedicated hardware and air-gapped "
            "management planes. Enterprises report an average 34 percent increase in cloud operating "
            "expenditure attributable to sovereignty compliance requirements. Hybrid cloud orchestration "
            "platforms that automate policy-driven workload placement have seen accelerated adoption."
        ),
    },
    {
        "id": 34,
        "title": "Liquid Cooling Becomes Standard in AI Data Centres",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "The explosive power density requirements of AI training clusters have made direct liquid "
            "cooling a de facto industry standard, with over 75 percent of new hyperscale data centre "
            "construction projects specifying immersion or direct-to-chip cooling infrastructure. "
            "Traditional air-cooled raised-floor designs are being retired rapidly as power-per-rack "
            "densities exceed the 30-kilowatt threshold that air-side economisers cannot efficiently "
            "manage. Leading facility operators are investing in on-site water recycling and heat "
            "recovery systems to offset the environmental footprint of cooling infrastructure."
        ),
    },
    {
        "id": 35,
        "title": "Edge AI Inference Chips Enter Volume Production",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "Three semiconductor companies simultaneously announced volume production of dedicated edge "
            "AI inference chips targeting industrial IoT, autonomous vehicle perception, and smart "
            "retail applications. The devices achieve sub-10-millisecond inference latency for "
            "computer-vision workloads while consuming less than 5 watts of power, enabling deployment "
            "on battery-constrained platforms. A standardised model interchange format supported by all "
            "three vendors has simplified the workflow for deploying trained cloud models to heterogeneous "
            "edge device fleets at scale."
        ),
    },
    {
        "id": 36,
        "title": "Multi-Cloud Cost Optimisation Becomes Board Priority",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "A survey of 800 enterprise CIOs found that unmanaged multi-cloud expenditure exceeded "
            "budget projections by an average of 38 percent, elevating cloud cost governance to a "
            "top-three board-level technology priority. FinOps platforms leveraging machine-learning "
            "recommendations for right-sizing compute instances, reserved capacity planning, and spot "
            "instance orchestration are delivering average savings of 24 percent on cloud bills. "
            "Tag-hygiene enforcement and showback reporting to individual business units are the most "
            "commonly implemented cost-visibility practices."
        ),
    },
    {
        "id": 37,
        "title": "Open-Source Silicon Initiative Challenges Proprietary GPU Duopoly",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "A consortium of technology companies launched an open-source GPU instruction set "
            "architecture and companion open-source firmware stack, aiming to challenge proprietary "
            "lock-in in the AI accelerator market. Early silicon tape-outs achieved 78 percent of "
            "the performance density of comparable commercial accelerators while offering fully "
            "auditable hardware and software stacks. Academic research groups and national laboratories "
            "expressed strong interest in the architecture as a foundation for specialised scientific "
            "computing workloads."
        ),
    },
    {
        "id": 38,
        "title": "Serverless AI Inference Cuts Latency for Real-Time Applications",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "Cloud providers have extended serverless computing paradigms to GPU-accelerated AI "
            "inference, offering sub-50-millisecond cold-start times for large language model serving "
            "through aggressive model pre-warming and predictive scheduling. Enterprise customers "
            "migrating from always-on GPU instances to serverless inference report cost reductions "
            "between 55 and 70 percent for workloads with intermittent traffic patterns. The model "
            "is particularly attractive for internal tooling and document-processing pipelines that "
            "experience burst demand during business hours."
        ),
    },
    {
        "id": 39,
        "title": "AI Workload Optimiser Reduces HPC Cluster Idle Time",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "A reinforcement-learning-based job scheduler deployed across a national high-performance "
            "computing facility reduced cluster idle time from 22 percent to under 7 percent over a "
            "three-month evaluation period by accurately predicting job runtimes and backfilling "
            "scheduling gaps. The system integrates with existing PBS and Slurm workload managers "
            "and learned scheduling policies from 18 months of historical job telemetry. The "
            "optimiser also reduced average job queue wait times by 41 percent, significantly "
            "improving researcher productivity."
        ),
    },
    {
        "id": 40,
        "title": "5G Private Networks Enable Smart Factory Transformation",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "A major automotive manufacturer completed deployment of a campus-wide 5G private network "
            "across its largest production facility, enabling wireless connectivity for over 4,000 "
            "autonomous guided vehicles and robotic assembly arms. Ultra-low-latency communication "
            "facilitated closed-loop machine-vision quality inspection with defect-detection latency "
            "below eight milliseconds. The transition from legacy Wi-Fi infrastructure eliminated "
            "connectivity dead zones and reduced unplanned production line stoppages by 63 percent "
            "during the first operational quarter."
        ),
    },
    {
        "id": 41,
        "title": "Memory Bandwidth Becomes AI Training Bottleneck",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "Industry analyses confirm that memory bandwidth, rather than floating-point compute "
            "throughput, is the primary bottleneck constraining the training efficiency of hundred-"
            "billion-parameter transformer models. Leading research teams have adopted mixed-precision "
            "training, activation checkpointing, and tensor parallelism strategies to partially "
            "mitigate the constraint. Semiconductor roadmaps for the next three years are prioritising "
            "HBM4 adoption and processing-in-memory architectures to relieve the memory-wall limitation "
            "for AI-centric workloads."
        ),
    },
    {
        "id": 42,
        "title": "Colocation Data Centres Surge on AI Power Demand",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "Third-party colocation data centre operators reported capacity utilisation rates exceeding "
            "95 percent across tier-one markets, driven by insatiable demand for AI training and "
            "inference infrastructure from cloud providers and large enterprises. Lease rates for "
            "powered shell space rose 28 percent year-on-year as new supply struggled to keep pace "
            "with construction permitting delays and grid interconnection queues averaging 36 months. "
            "Operators with existing power purchase agreements for renewable energy commanded "
            "significant premium pricing in sustainability-focused enterprise procurement processes."
        ),
    },
    {
        "id": 43,
        "title": "Photonic Computing Chips Achieve Commercial Milestone",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "A photonic computing startup achieved a commercial milestone by demonstrating matrix "
            "multiplication operations at speeds 100 times faster than equivalent electronic "
            "implementations with 10 times lower energy consumption for specific AI inference "
            "workloads. The breakthrough leverages silicon photonics waveguide arrays to perform "
            "analog computation using coherent light rather than electron flow. Adoption remains "
            "constrained by the need for analog-to-digital conversion at system boundaries, but "
            "the team published a roadmap for hybrid photonic-electronic co-processors."
        ),
    },
    {
        "id": 44,
        "title": "Zero Trust Architecture Mandate for Federal Cloud Workloads",
        "category": "Enterprise Cloud & Hardware",
        "label": 2,
        "text": (
            "The Office of Management and Budget issued updated guidance mandating zero-trust network "
            "architecture for all federal agency cloud workloads by the end of the fiscal year. "
            "Agencies must implement continuous identity verification, encrypted micro-segmented "
            "network paths, and behavioural analytics for anomaly detection across all cloud-hosted "
            "applications. Compliance assessments will be conducted quarterly, with non-compliant "
            "agencies subject to funding restrictions. Cloud service providers holding FedRAMP "
            "authorisations must provide compliance evidence packages within 90 days."
        ),
    },

    # ─────────────────────────────────────────────────────────────
    # Category 3 : Policy & Regulation
    # ─────────────────────────────────────────────────────────────
    {
        "id": 45,
        "title": "EU AI Act Enforcement Begins for High-Risk Systems",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "The European Union's AI Act entered its enforcement phase for high-risk AI applications "
            "in critical infrastructure, employment, and essential services, requiring conformity "
            "assessments, technical documentation, and registration in the EU AI database. Companies "
            "deploying non-compliant high-risk systems face fines of up to 30 million euros or "
            "6 percent of global annual turnover. The European AI Office received over 2,400 "
            "compliance inquiries in the first week, highlighting widespread uncertainty about "
            "risk classification criteria and conformity assessment procedures."
        ),
    },
    {
        "id": 46,
        "title": "US Executive Order on AI Safety Mandates Red-Teaming Reports",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "A presidential executive order on AI safety requires developers of frontier AI models "
            "exceeding specific training compute thresholds to submit pre-deployment safety evaluations, "
            "red-teaming reports, and incident disclosures to the Department of Commerce within 90 "
            "days of model completion. The order also establishes an AI Safety Institute with authority "
            "to develop testing standards and coordinate with international counterparts. Technology "
            "companies expressed support for the transparency requirements while seeking clearer "
            "definitions of the compute thresholds that trigger mandatory obligations."
        ),
    },
    {
        "id": 47,
        "title": "GDPR Enforcement Action Targets AI Training Data Practices",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "The Irish Data Protection Commission levied a 1.2-billion-euro fine against a US "
            "technology company for systematically using European users' personal data to train "
            "commercial AI models without adequate legal basis under GDPR. The enforcement decision "
            "requires deletion of all model parameters derived from unlawfully processed data within "
            "18 months. The ruling sets a precedent that could affect training practices across the "
            "AI industry and prompted emergency legal reviews at dozens of major AI developers "
            "regarding their data provenance documentation."
        ),
    },
    {
        "id": 48,
        "title": "G7 Leaders Adopt Hiroshima AI Process Principles",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "Leaders from the Group of Seven nations endorsed an expanded set of AI governance "
            "principles under the Hiroshima AI Process, including voluntary commitments to "
            "transparency reporting, watermarking of AI-generated content, and incident-sharing "
            "frameworks for safety-relevant model failures. The principles also address risks from "
            "dual-use AI systems capable of contributing to weapons of mass destruction development. "
            "Civil society organisations criticised the voluntary nature of the commitments, arguing "
            "that binding international treaty obligations are necessary to ensure compliance."
        ),
    },
    {
        "id": 49,
        "title": "Digital Markets Act Reshapes Platform Business Models",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "The European Commission's Digital Markets Act gatekeeper obligations entered full force, "
            "requiring dominant platform operators to allow third-party app stores, interoperability "
            "with rival messaging services, and data portability on commercially reasonable terms. "
            "Non-compliance investigations were opened against three major technology companies within "
            "the first month. Legal experts anticipate multi-year enforcement proceedings and expected "
            "platform business model restructuring worth an estimated 40 billion dollars in divested "
            "or modified revenue streams."
        ),
    },
    {
        "id": 50,
        "title": "India's Digital Personal Data Protection Act Regulations Finalised",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "India's Ministry of Electronics and Information Technology released implementing "
            "regulations under the Digital Personal Data Protection Act, establishing data principal "
            "rights to access, correction, and erasure of personal data held by significant data "
            "fiduciaries. Consent managers must obtain granular purpose-specific consents before "
            "any processing. Cross-border data transfers to approved countries will require standard "
            "contractual clauses and periodic compliance audits. Non-compliant organisations face "
            "penalties of up to 250 crore rupees per violation."
        ),
    },
    {
        "id": 51,
        "title": "Autonomous Weapons Treaty Negotiations Stall at UN",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "United Nations Conference on Certain Conventional Weapons negotiations on a binding "
            "treaty to prohibit fully autonomous lethal weapons systems collapsed after six major "
            "military powers declined to accept language that would mandate meaningful human control "
            "over targeting decisions. Humanitarian organisations condemned the impasse as a dangerous "
            "regulatory vacuum. The failure to agree has accelerated unilateral autonomous weapons "
            "development programmes in multiple nations and raised concerns among international law "
            "scholars about accountability frameworks for AI-enabled military systems."
        ),
    },
    {
        "id": 52,
        "title": "FTC Challenges AI Company Merger on Antitrust Grounds",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "The Federal Trade Commission filed to block a proposed 28-billion-dollar acquisition of "
            "a leading AI model developer by a hyperscale cloud provider, arguing that the transaction "
            "would entrench the acquirer's dominance across foundation model development, inference "
            "infrastructure, and enterprise AI application markets. The lawsuit alleges that exclusive "
            "compute supply agreements bundled with the deal would foreclose competition for rival "
            "model developers. The case is widely seen as a landmark test of antitrust doctrine's "
            "applicability to AI platform ecosystems."
        ),
    },
    {
        "id": 53,
        "title": "Copyright Litigation Shapes AI Training Data Precedents",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "A federal appeals court affirmed that AI companies must obtain licences for copyrighted "
            "material used in training datasets, rejecting fair-use defences in a landmark case "
            "brought by a coalition of authors and journalists. The ruling requires the defendant "
            "to disclose its full training data provenance and negotiate royalty frameworks with "
            "rights holders within 12 months. Publishers, record labels, and visual arts organisations "
            "filed a wave of new lawsuits following the decision, seeking to establish industry-wide "
            "licensing standards for generative AI training."
        ),
    },
    {
        "id": 54,
        "title": "National AI Strategy Allocates 20 Billion in Public Research Funding",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "A federal government unveiled a comprehensive national AI strategy allocating 20 billion "
            "dollars in public research funding over five years to establish domestic AI research "
            "institutes, fund graduate training programmes, and support compute infrastructure at "
            "universities. The strategy prioritises sovereign AI capabilities in sectors deemed "
            "strategically critical including defence, health, and climate modelling. International "
            "collaboration provisions permit joint research with allied nations while restricting "
            "sensitive technology transfer to designated adversarial countries."
        ),
    },
    {
        "id": 55,
        "title": "Biometric Data Regulation Tightens Across US States",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "Fourteen US states enacted biometric privacy laws modelled on Illinois' Biometric "
            "Information Privacy Act, imposing written consent, storage limitation, and destruction "
            "requirements on organisations collecting facial recognition, fingerprint, and iris scan "
            "data. Class-action litigation under the Illinois statute has already resulted in "
            "settlements exceeding three billion dollars. Technology companies operating facial "
            "recognition products have withdrawn from several state markets rather than bear "
            "litigation exposure, and a federal preemptive framework remains stalled in Congress."
        ),
    },
    {
        "id": 56,
        "title": "AI Liability Directive Proposed in European Parliament",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "The European Parliament's legal affairs committee advanced a draft AI Liability Directive "
            "that would establish a rebuttable presumption of causation for harm caused by high-risk "
            "AI systems, effectively reversing the burden of proof onto defendants. Affected companies "
            "must maintain comprehensive audit logs enabling reconstruction of AI decision pathways. "
            "Insurers are developing new product lines to underwrite AI liability exposure as the "
            "legislative process moves toward trilogue negotiations. Legal scholars debate whether "
            "the strict-liability approach will stifle AI innovation across the continent."
        ),
    },
    {
        "id": 57,
        "title": "Children's Online Safety Act Imposes AI Content Obligations",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "Comprehensive children's online safety legislation enacted age-appropriate design codes "
            "requiring platforms to deploy AI-powered age verification, restrict algorithmic "
            "recommendation systems for minor users, and conduct annual data-impact assessments. "
            "Social media companies must implement parental supervision tools and are prohibited "
            "from serving behavioural advertising to users under the age of 18. Independent auditors "
            "will assess platform compliance annually with findings reported to the designated "
            "national regulatory authority."
        ),
    },
    {
        "id": 58,
        "title": "Open-Source AI Model Licensing Debate Intensifies",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "Policymakers and AI researchers are engaged in a contentious debate over whether "
            "powerful open-source AI model releases should require use-restriction licences that "
            "prohibit applications in weapons development, mass surveillance, and disinformation "
            "campaigns. Proponents argue that unrestricted open release of frontier models creates "
            "catastrophic dual-use risks that responsible disclosure norms cannot adequately address. "
            "Open-source advocates counter that licensing restrictions are unenforceable and will "
            "disproportionately disadvantage academic researchers and smaller enterprises."
        ),
    },
    {
        "id": 59,
        "title": "AI Election Integrity Rules Adopted Before Major Votes",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "Electoral commissions in twelve democracies adopted emergency AI election integrity "
            "rules requiring mandatory disclosure labels on AI-generated political advertisements, "
            "banning synthetic voice impersonation of candidates, and establishing rapid-response "
            "fact-checking partnerships with social media platforms. Enforcement challenges emerged "
            "immediately as detection tools struggled to keep pace with the sophistication of "
            "AI-generated content. Civil liberties groups raised constitutional questions about "
            "compelled speech obligations imposed on political campaigns using AI-assisted messaging."
        ),
    },
    {
        "id": 60,
        "title": "Algorithmic Accountability Bill Requires Audit Trails",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "Federal legislation requiring organisations deploying consequential automated decision "
            "systems to maintain machine-readable audit trails, conduct annual algorithmic impact "
            "assessments, and establish meaningful human-review mechanisms passed the Senate Commerce "
            "Committee. The bill covers AI systems influencing credit, housing, employment, healthcare, "
            "and education decisions. Advocacy groups lauded the accountability provisions while "
            "technology industry representatives sought exemptions for lower-stakes applications "
            "and clearer definitional boundaries around the term 'consequential'."
        ),
    },
    {
        "id": 61,
        "title": "International AI Safety Summit Launches Incident Registry",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "Delegates from 28 nations at the International AI Safety Summit agreed to establish a "
            "confidential AI incident registry modelled on aviation's safety reporting systems, "
            "enabling voluntary submission of near-miss events and safety-relevant model failures "
            "without triggering immediate regulatory action. The registry will be operated by an "
            "independent secretariat and produce de-identified trend reports for public release. "
            "Participating nations committed to coordinating export controls on advanced AI training "
            "hardware to prevent circumvention of domestic safety requirements."
        ),
    },
    {
        "id": 62,
        "title": "AI Transparency Report Mandate for Public Procurement",
        "category": "Policy & Regulation",
        "label": 3,
        "text": (
            "New federal procurement rules require AI vendors supplying government agencies to "
            "publish quarterly transparency reports disclosing model architecture summaries, "
            "training data categories, fairness evaluation results across demographic groups, "
            "and documented known limitations. Agencies are prohibited from procuring AI systems "
            "that cannot provide model cards meeting standardised disclosure requirements. The rules "
            "are expected to catalyse broader industry adoption of voluntary transparency practices "
            "as vendors align documentation processes to meet government customer requirements."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────

def get_all_texts():
    """Return list of (text, label_int) tuples for the whole corpus."""
    return [(item["text"], item["label"]) for item in CORPUS]


def get_texts_by_category(category: str):
    """Return all corpus entries matching a given category string."""
    return [item for item in CORPUS if item["category"] == category]


def get_sample_titles():
    """Return list of display strings 'ID: title (category)' for UI dropdown."""
    return [f'{item["id"]:02d}. {item["title"]} [{item["category"]}]' for item in CORPUS]


def get_item_by_index(idx: int):
    """Return a corpus item by its list position."""
    return CORPUS[idx]
