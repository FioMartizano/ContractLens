import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";

type RiskLevel = "high" | "medium" | "low";

type ContractClause = {
  id: string;
  page_number: number;
  clause_type: string;
  text: string;
  party_obligated: string;
  deadlines: string[];
  money_amounts: string[];
  risk_score: number;
  risk_level: RiskLevel;
  explanation: string;
  deviation: string;
  suggested_rewrite: string;
  standard_reference: {
    title: string;
    text: string;
    distance: number;
  } | null;
};

type AnalysisResponse = {
  document_id: string;
  filename: string;
  file_type: string;
  total_pages: number;
  total_characters: number;
  chunks_indexed: number;
  uploaded_at: string;
  overall_score: number;
  risk_counts: Record<RiskLevel, number>;
  clauses: ContractClause[];
  disclaimer: string;
};

type ComparisonResponse = {
  first_document: { document_id: string; filename: string; overall_score: number };
  second_document: { document_id: string; filename: string; overall_score: number };
  changes: {
    first_clause: ContractClause | null;
    second_clause: ContractClause | null;
    risk_change: "increased" | "decreased" | "unchanged" | "added" | "removed";
    explanation: string;
  }[];
  disclaimer: string;
};

type SearchResult = {
  text: string;
  metadata: { document_id: string; filename: string; page: number };
  distance: number;
};

const API_BASE_URL = (import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000").replace(
  /\/$/,
  "",
);

function isContractClause(value: unknown): value is ContractClause {
  return (
    typeof value === "object" &&
    value !== null &&
    "id" in value &&
    typeof value.id === "string" &&
    "page_number" in value &&
    typeof value.page_number === "number" &&
    "clause_type" in value &&
    typeof value.clause_type === "string" &&
    "text" in value &&
    typeof value.text === "string" &&
    "party_obligated" in value &&
    typeof value.party_obligated === "string" &&
    "deadlines" in value &&
    Array.isArray(value.deadlines) &&
    value.deadlines.every((deadline) => typeof deadline === "string") &&
    "money_amounts" in value &&
    Array.isArray(value.money_amounts) &&
    value.money_amounts.every((amount) => typeof amount === "string") &&
    "risk_score" in value &&
    typeof value.risk_score === "number" &&
    "risk_level" in value &&
    (value.risk_level === "high" || value.risk_level === "medium" || value.risk_level === "low") &&
    "explanation" in value &&
    typeof value.explanation === "string" &&
    "deviation" in value &&
    typeof value.deviation === "string" &&
    "suggested_rewrite" in value &&
    typeof value.suggested_rewrite === "string" &&
    "standard_reference" in value &&
    (value.standard_reference === null ||
      (typeof value.standard_reference === "object" &&
        value.standard_reference !== null &&
        "title" in value.standard_reference &&
        typeof value.standard_reference.title === "string" &&
        "text" in value.standard_reference &&
        typeof value.standard_reference.text === "string" &&
        "distance" in value.standard_reference &&
        typeof value.standard_reference.distance === "number"))
  );
}

function isAnalysisResponse(value: unknown): value is AnalysisResponse {
  return (
    typeof value === "object" &&
    value !== null &&
    "document_id" in value &&
    typeof value.document_id === "string" &&
    "filename" in value &&
    typeof value.filename === "string" &&
    "file_type" in value &&
    typeof value.file_type === "string" &&
    "total_pages" in value &&
    typeof value.total_pages === "number" &&
    "total_characters" in value &&
    typeof value.total_characters === "number" &&
    "chunks_indexed" in value &&
    typeof value.chunks_indexed === "number" &&
    "uploaded_at" in value &&
    typeof value.uploaded_at === "string" &&
    "overall_score" in value &&
    typeof value.overall_score === "number" &&
    "risk_counts" in value &&
    typeof value.risk_counts === "object" &&
    value.risk_counts !== null &&
    "high" in value.risk_counts &&
    typeof value.risk_counts.high === "number" &&
    "medium" in value.risk_counts &&
    typeof value.risk_counts.medium === "number" &&
    "low" in value.risk_counts &&
    typeof value.risk_counts.low === "number" &&
    "clauses" in value &&
    Array.isArray(value.clauses) &&
    value.clauses.every(isContractClause) &&
    "disclaimer" in value &&
    typeof value.disclaimer === "string"
  );
}

function isAnalysisListResponse(value: unknown): value is { documents: AnalysisResponse[] } {
  return (
    typeof value === "object" &&
    value !== null &&
    "documents" in value &&
    Array.isArray(value.documents) &&
    value.documents.every(isAnalysisResponse)
  );
}

function isComparisonResponse(value: unknown): value is ComparisonResponse {
  return (
    typeof value === "object" &&
    value !== null &&
    "first_document" in value &&
    typeof value.first_document === "object" &&
    value.first_document !== null &&
    "document_id" in value.first_document &&
    typeof value.first_document.document_id === "string" &&
    "filename" in value.first_document &&
    typeof value.first_document.filename === "string" &&
    "overall_score" in value.first_document &&
    typeof value.first_document.overall_score === "number" &&
    "second_document" in value &&
    typeof value.second_document === "object" &&
    value.second_document !== null &&
    "document_id" in value.second_document &&
    typeof value.second_document.document_id === "string" &&
    "filename" in value.second_document &&
    typeof value.second_document.filename === "string" &&
    "overall_score" in value.second_document &&
    typeof value.second_document.overall_score === "number" &&
    "changes" in value &&
    Array.isArray(value.changes) &&
    value.changes.every(
      (change) =>
        typeof change === "object" &&
        change !== null &&
        "first_clause" in change &&
        (change.first_clause === null || isContractClause(change.first_clause)) &&
        "second_clause" in change &&
        (change.second_clause === null || isContractClause(change.second_clause)) &&
        "risk_change" in change &&
        ["increased", "decreased", "unchanged", "added", "removed"].includes(
          String(change.risk_change),
        ) &&
        "explanation" in change &&
        typeof change.explanation === "string",
    ) &&
    "disclaimer" in value &&
    typeof value.disclaimer === "string"
  );
}

function isSearchResponse(value: unknown): value is { results: SearchResult[] } {
  return (
    typeof value === "object" &&
    value !== null &&
    "results" in value &&
    Array.isArray(value.results) &&
    value.results.every(
      (result) =>
        typeof result === "object" &&
        result !== null &&
        "text" in result &&
        typeof result.text === "string" &&
        "distance" in result &&
        typeof result.distance === "number" &&
        "metadata" in result &&
        typeof result.metadata === "object" &&
        result.metadata !== null &&
        "document_id" in result.metadata &&
        typeof result.metadata.document_id === "string" &&
        "filename" in result.metadata &&
        typeof result.metadata.filename === "string" &&
        "page" in result.metadata &&
        typeof result.metadata.page === "number",
    )
  );
}

function getApiError(payload: unknown, fallback: string): string {
  return typeof payload === "object" &&
    payload !== null &&
    "detail" in payload &&
    typeof payload.detail === "string"
    ? payload.detail
    : fallback;
}

type IconName =
  | "brand"
  | "grid"
  | "file"
  | "compare"
  | "book"
  | "settings"
  | "help"
  | "bell"
  | "search"
  | "upload"
  | "chevron"
  | "arrow"
  | "download"
  | "more"
  | "check"
  | "clock"
  | "alert"
  | "shield"
  | "sparkles"
  | "close"
  | "plus";

function Icon({
  name,
  size = 20,
  className = "",
}: {
  name: IconName;
  size?: number;
  className?: string;
}) {
  const paths: Record<IconName, ReactNode> = {
    brand: (
      <>
        <path d="M12 3 4.5 6v5.2c0 4.8 3.2 8.8 7.5 9.8 4.3-1 7.5-5 7.5-9.8V6L12 3Z" />
        <path d="m8.8 12 2 2 4.5-4.7" />
      </>
    ),
    grid: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="2" />
        <rect x="14" y="3" width="7" height="7" rx="2" />
        <rect x="3" y="14" width="7" height="7" rx="2" />
        <rect x="14" y="14" width="7" height="7" rx="2" />
      </>
    ),
    file: (
      <>
        <path d="M6 2.8h8l4 4V21H6z" />
        <path d="M14 3v5h4M9 12h6M9 16h6" />
      </>
    ),
    compare: (
      <>
        <path d="M8 7h11M16 4l3 3-3 3M16 17H5M8 14l-3 3 3 3" />
      </>
    ),
    book: (
      <>
        <path d="M4 4.5A3.5 3.5 0 0 1 7.5 8H20v12H7.5A3.5 3.5 0 0 0 4 16.5z" />
        <path d="M4 4.5v12M8 12h8M8 16h6" />
      </>
    ),
    settings: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1a1.7 1.7 0 0 0 1.9.3A1.7 1.7 0 0 0 10 3v-.2h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z" />
      </>
    ),
    help: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M9.6 9a2.5 2.5 0 1 1 3.2 2.4c-.8.3-.8 1-.8 1.6M12 17h.01" />
      </>
    ),
    bell: (
      <>
        <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 8.5h18C21 15 18 15 18 8Z" />
        <path d="M10 20h4" />
      </>
    ),
    search: (
      <>
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-4-4" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V4M7.5 8.5 12 4l4.5 4.5" />
        <path d="M5 14v5h14v-5" />
      </>
    ),
    chevron: <path d="m9 18 6-6-6-6" />,
    arrow: <path d="m5 12 14 0m-5-5 5 5-5 5" />,
    download: (
      <>
        <path d="M12 3v12m-4-4 4 4 4-4M5 20h14" />
      </>
    ),
    more: (
      <>
        <circle cx="5" cy="12" r="1" fill="currentColor" stroke="none" />
        <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
        <circle cx="19" cy="12" r="1" fill="currentColor" stroke="none" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    alert: (
      <>
        <path d="M10.3 3.8 2.5 18a2 2 0 0 0 1.8 3h15.4a2 2 0 0 0 1.8-3L13.7 3.8a2 2 0 0 0-3.4 0Z" />
        <path d="M12 9v4M12 17h.01" />
      </>
    ),
    shield: (
      <>
        <path d="M12 3 5 6v5c0 4.6 2.9 8.2 7 9.5 4.1-1.3 7-4.9 7-9.5V6z" />
        <path d="m9 12 2 2 4-4" />
      </>
    ),
    sparkles: (
      <>
        <path d="m12 3 1.1 3.2L16 8l-2.9 1.8L12 13l-1.1-3.2L8 8l2.9-1.8zM18.5 14l.7 2.3 2.3.7-2.3.7-.7 2.3-.7-2.3-2.3-.7 2.3-.7zM5.5 12l.7 2.3 2.3.7-2.3.7L5.5 18l-.7-2.3-2.3-.7 2.3-.7z" />
      </>
    ),
    close: <path d="m6 6 12 12M18 6 6 18" />,
    plus: <path d="M12 5v14M5 12h14" />,
  };
  return (
    <svg
      aria-hidden="true"
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}

const EMPTY_CLAUSES: ContractClause[] = [];
const RISK_FILTERS: ("all" | RiskLevel)[] = ["all", "high", "medium", "low"];

const navItems: { label: string; icon: IconName; id: string }[] = [
  { label: "Overview", icon: "grid", id: "overview" },
  { label: "Documents", icon: "file", id: "documents" },
  { label: "Compare", icon: "compare", id: "compare" },
  { label: "Clause library", icon: "book", id: "library" },
];

function RiskBadge({ level }: { level: string }) {
  return (
    <span className={`risk-badge ${level}`}>
      <span />
      {level[0].toUpperCase() + level.slice(1)} risk
    </span>
  );
}

function UploadModal({
  onClose,
  onUpload,
  isUploading,
  uploadError,
  uploadSuccess,
}: {
  onClose: () => void;
  onUpload: (file: File) => Promise<void>;
  isUploading: boolean;
  uploadError: string | null;
  uploadSuccess: string | null;
}) {
  const [file, setFile] = useState<File | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (!isUploading) {
      setElapsedSeconds(0);
      return;
    }

    const interval = window.setInterval(() => {
      setElapsedSeconds((seconds) => seconds + 1);
    }, 1000);

    return () => window.clearInterval(interval);
  }, [isUploading]);

  const loadingStage =
    elapsedSeconds < 15
      ? "Uploading and extracting document text"
      : elapsedSeconds < 45
        ? "Extracting clauses and identifying source quotes"
        : elapsedSeconds < 90
          ? "Comparing clauses with the ChromaDB library"
          : "Running final Gemini risk and rewrite analysis";

  const submitUpload = () => {
    if (file) void onUpload(file);
  };

  const requestClose = () => {
    if (!isUploading) onClose();
  };

  return (
    <div className="modal-backdrop" onMouseDown={requestClose}>
      <div className="modal" onMouseDown={(e) => e.stopPropagation()}>
        <button
          type="button"
          className="icon-button modal-close"
          onClick={requestClose}
          disabled={isUploading}
        >
          <Icon name="close" size={18} />
        </button>
        <div className="modal-icon">
          <Icon name="upload" />
        </div>
        <h2>Upload a contract</h2>
        <p>We’ll analyze each clause and highlight terms that deserve your attention.</p>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.jpg,.jpeg,.png"
          style={{ display: "none" }}
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button
          className={`dropzone ${file ? "has-file" : ""}`}
          onClick={() => fileRef.current?.click()}
          type="button"
        >
          <span className="drop-icon">
            <Icon name={file ? "check" : "file"} />
          </span>
          <strong>{file?.name ?? "Choose a file"}</strong>
          <small>PDF, JPG, or PNG · Maximum 10 MB</small>
        </button>
        {uploadError && <p role="alert">{uploadError}</p>}
        {uploadSuccess && <p role="status">{uploadSuccess}</p>}
        {isUploading && (
          <div className="analysis-loader" role="status" aria-live="polite">
            <span className="analysis-spinner" aria-hidden="true" />
            <div>
              <strong>{loadingStage}</strong>
              <small>
                Processing {elapsedSeconds}s · Please keep this window open
              </small>
            </div>
          </div>
        )}
        <div className="privacy-note">
          <Icon name="shield" size={17} />
          <span>
            Contract text is sent to your configured Gemini API and indexed locally in ChromaDB.
          </span>
        </div>
        <button
          className="primary-button full"
          disabled={!file || isUploading}
          onClick={submitUpload}
        >
          {isUploading
            ? "Analyzing and indexing..."
            : uploadSuccess
              ? "Analysis complete"
              : "Analyze contract"}
        </button>
      </div>
    </div>
  );
}

function CompareView({
  documents,
  firstDocumentId,
  secondDocumentId,
  onSelectFirst,
  onSelectSecond,
  onCompare,
  comparison,
  isComparing,
  error,
}: {
  documents: AnalysisResponse[];
  firstDocumentId: string;
  secondDocumentId: string;
  onSelectFirst: (id: string) => void;
  onSelectSecond: (id: string) => void;
  onCompare: () => void;
  comparison: ComparisonResponse | null;
  isComparing: boolean;
  error: string | null;
}) {
  return (
    <main className="page-content">
      <div className="page-heading">
        <div>
          <p className="eyebrow">CONTRACT COMPARISON</p>
          <h1>Compare agreements</h1>
          <p>See exactly what changed between two versions and how it affects your risk.</p>
        </div>
      </div>
      <section className="compare-card">
        <div className="compare-intro">
          <span className="feature-icon purple">
            <Icon name="compare" />
          </span>
          <h2>Spot meaningful changes, not just different words</h2>
          <p>
            ContractLens compares clause meaning, obligations, and risk—so a subtle change doesn’t
            slip past you.
          </p>
        </div>
        <div className="compare-slots">
          <label className="compare-slot">
            <span>
              <Icon name="plus" />
            </span>
            <strong>Original contract</strong>
            <select
              aria-label="Original contract"
              value={firstDocumentId}
              onChange={(event) => onSelectFirst(event.target.value)}
            >
              <option value="">Select an analyzed contract</option>
              {documents.map((document) => (
                <option key={document.document_id} value={document.document_id}>
                  {document.filename}
                </option>
              ))}
            </select>
          </label>
          <div className="versus">VS</div>
          <label className="compare-slot">
            <span>
              <Icon name="plus" />
            </span>
            <strong>Revised contract</strong>
            <select
              aria-label="Revised contract"
              value={secondDocumentId}
              onChange={(event) => onSelectSecond(event.target.value)}
            >
              <option value="">Select an analyzed contract</option>
              {documents.map((document) => (
                <option key={document.document_id} value={document.document_id}>
                  {document.filename}
                </option>
              ))}
            </select>
          </label>
        </div>
        {documents.length < 2 && (
          <p className="compare-hint">Upload at least two contracts before comparing them.</p>
        )}
        {error && <p className="compare-error" role="alert">{error}</p>}
        <button
          className="primary-button compare-action"
          disabled={
            isComparing ||
            !firstDocumentId ||
            !secondDocumentId ||
            firstDocumentId === secondDocumentId
          }
          onClick={onCompare}
        >
          {isComparing ? "Comparing..." : "Compare contracts"}
        </button>
      </section>
      {comparison && (
        <section className="comparison-results" aria-live="polite">
          <h2>
            {comparison.first_document.filename} <span>vs.</span>{" "}
            {comparison.second_document.filename}
          </h2>
          {comparison.changes.map((change, index) => (
            <article key={`${change.first_clause?.id ?? "removed"}-${change.second_clause?.id ?? "added"}-${index}`}>
              <div>
                <strong>{change.risk_change.replace("_", " ")} risk</strong>
                <p>{change.explanation}</p>
              </div>
              <div className="comparison-clause-pair">
                <p>{change.first_clause?.text ?? "No matching clause in the original contract."}</p>
                <p>{change.second_clause?.text ?? "No matching clause in the revised contract."}</p>
              </div>
            </article>
          ))}
          <small>{comparison.disclaimer}</small>
        </section>
      )}
      <section className="info-grid">
        <article>
          <span>01</span>
          <h3>Clause-by-clause matching</h3>
          <p>We match corresponding clauses even when they’ve been reordered or rewritten.</p>
        </article>
        <article>
          <span>02</span>
          <h3>Risk change tracking</h3>
          <p>See which changes make your agreement safer—and which introduce new exposure.</p>
        </article>
        <article>
          <span>03</span>
          <h3>Plain-language summary</h3>
          <p>Understand the practical impact of every important change without legal jargon.</p>
        </article>
      </section>
    </main>
  );
}

function App() {
  const [activeNav, setActiveNav] = useState("overview");
  const [activeClause, setActiveClause] = useState("");
  const [filter, setFilter] = useState("all");
  const [showUpload, setShowUpload] = useState(false);
  const [showMobileNav, setShowMobileNav] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [documents, setDocuments] = useState<AnalysisResponse[]>([]);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [firstDocumentId, setFirstDocumentId] = useState("");
  const [secondDocumentId, setSecondDocumentId] = useState("");
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [isComparing, setIsComparing] = useState(false);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [copyError, setCopyError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[] | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const clauses = analysis?.clauses ?? EMPTY_CLAUSES;
  const current = clauses.find((clause) => clause.id === activeClause) ?? clauses[0] ?? null;
  const overallRiskLevel: RiskLevel =
    analysis && analysis.overall_score <= 30
      ? "high"
      : analysis && analysis.overall_score <= 65
        ? "medium"
        : "low";
  const visibleClauses = useMemo(
    () =>
      filter === "all"
        ? clauses
        : clauses.filter((clause) => clause.risk_level === filter),
    [clauses, filter],
  );

  useEffect(() => {
    let cancelled = false;
    const loadDocuments = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/documents`);
        const payload: unknown = await response.json();
        if (!response.ok) {
          throw new Error(getApiError(payload, "Could not load analyzed contracts."));
        }
        if (!isAnalysisListResponse(payload)) {
          throw new Error("The server returned an invalid document list.");
        }
        if (cancelled) return;

        setDocuments((existing) => [
          ...payload.documents,
          ...existing.filter(
            (document) =>
              !payload.documents.some((saved) => saved.document_id === document.document_id),
          ),
        ]);
        setAnalysis((current) => current ?? payload.documents[0] ?? null);
        setActiveClause((current) => current || payload.documents[0]?.clauses[0]?.id || "");
        setFirstDocumentId((current) => current || payload.documents[0]?.document_id || "");
        setSecondDocumentId(
          (current) =>
            current ||
            payload.documents.find(
              (document) => document.document_id !== payload.documents[0]?.document_id,
            )?.document_id ||
            "",
        );
        setDocumentsError(null);
      } catch (error) {
        if (!cancelled) {
          setDocumentsError(
            error instanceof Error ? error.message : "Could not load analyzed contracts.",
          );
        }
      }
    };

    void loadDocuments();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleUpload = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    setIsUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15 * 60 * 1000);

    try {
      const response = await fetch(`${API_BASE_URL}/api/documents`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      const payload: unknown = await response.json();

      if (!response.ok) {
        const detail =
          typeof payload === "object" &&
          payload !== null &&
          "detail" in payload &&
          typeof payload.detail === "string"
            ? payload.detail
            : "Document upload failed.";
        throw new Error(detail);
      }

      if (!isAnalysisResponse(payload)) {
        throw new Error("The server returned an invalid analysis response.");
      }

      setAnalysis(payload);
      setDocuments((existing) => [
        payload,
        ...existing.filter((document) => document.document_id !== payload.document_id),
      ]);
      setActiveClause(payload.clauses[0]?.id ?? "");
      setActiveNav("overview");
      setFirstDocumentId((selected) => selected || payload.document_id);
      setSecondDocumentId((selected) =>
        selected && selected !== payload.document_id
          ? selected
          : firstDocumentId && firstDocumentId !== payload.document_id
            ? payload.document_id
            : documents.find((document) => document.document_id !== payload.document_id)
                ?.document_id ?? "",
      );
      setComparison(null);
      setDocumentsError(null);
      setUploadSuccess(
        `${payload.filename} analyzed: ${payload.clauses.length} clauses, ${payload.chunks_indexed} searchable sections.`,
      );
    } catch (error) {
      setUploadError(
        error instanceof DOMException && error.name === "AbortError"
          ? "Analysis timed out after 15 minutes. The backend may still be processing; check it before retrying."
          : error instanceof Error
            ? error.message
            : "Document upload failed.",
      );
    } finally {
      window.clearTimeout(timeout);
      setIsUploading(false);
    }
  };

  const handleCompare = async () => {
    if (!firstDocumentId || !secondDocumentId || firstDocumentId === secondDocumentId) return;

    setIsComparing(true);
    setComparisonError(null);
    setComparison(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          first_document_id: firstDocumentId,
          second_document_id: secondDocumentId,
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) throw new Error(getApiError(payload, "Contract comparison failed."));
      if (!isComparisonResponse(payload)) {
        throw new Error("The server returned an invalid comparison response.");
      }
      setComparison(payload);
    } catch (error) {
      setComparisonError(error instanceof Error ? error.message : "Contract comparison failed.");
    } finally {
      setIsComparing(false);
    }
  };

  const handleSearch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const query = searchQuery.trim();
    if (!query) return;

    setIsSearching(true);
    setSearchError(null);
    setSearchResults(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) throw new Error(getApiError(payload, "Document search failed."));
      if (!isSearchResponse(payload)) {
        throw new Error("The server returned invalid search results.");
      }
      setSearchResults(payload.results);
    } catch (error) {
      setSearchError(error instanceof Error ? error.message : "Document search failed.");
    } finally {
      setIsSearching(false);
    }
  };

  const openSearchResult = (result: SearchResult) => {
    const document = documents.find(
      (item) => item.document_id === result.metadata.document_id,
    );
    if (document) {
      setAnalysis(document);
      setActiveNav("overview");
      setActiveClause(
        document.clauses.find((clause) => clause.page_number === result.metadata.page)?.id ??
          document.clauses[0]?.id ??
          "",
      );
    }
    setSearchResults(null);
  };

  const copyRewrite = async (text: string) => {
    setCopyError(null);
    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      setCopyError(error instanceof Error ? error.message : "Could not copy the suggested wording.");
    }
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${showMobileNav ? "open" : ""}`}>
        <div className="logo">
          <span>
            <Icon name="brand" />
          </span>
          <strong>
            Contract<span>Lens</span>
          </strong>
        </div>
        <nav>
          <p>WORKSPACE</p>
          {navItems.map((item) => (
            <button
              key={item.id}
              className={activeNav === item.id ? "active" : ""}
              onClick={() => {
                setActiveNav(item.id);
                setShowMobileNav(false);
              }}
            >
              <Icon name={item.icon} size={19} />
              <span>{item.label}</span>
              {item.id === "documents" && <b>{documents.length}</b>}
            </button>
          ))}
        </nav>
        <div className="side-bottom">
          <button>
            <Icon name="help" size={19} />
            Help & support
          </button>
          <button>
            <Icon name="settings" size={19} />
            Settings
          </button>
          <div className="profile">
            <div className="avatar">AM</div>
            <div>
              <strong>Alex Morgan</strong>
              <small>Free plan</small>
            </div>
            <Icon name="more" size={18} />
          </div>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setShowMobileNav((v) => !v)}>
            <span />
            <span />
            <span />
          </button>
          <div className="search-container">
            <form className="search" onSubmit={handleSearch}>
              <Icon name="search" size={18} />
              <input
                aria-label="Search contract clauses"
                placeholder="Search contract clauses..."
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
              />
              <kbd>{isSearching ? "..." : "Enter"}</kbd>
            </form>
            {searchError && <p className="search-message" role="alert">{searchError}</p>}
            {searchResults && (
              <div className="search-results" role="region" aria-label="Contract search results">
                {searchResults.length === 0 ? (
                  <p>No matching contract sections found.</p>
                ) : (
                  searchResults.map((result, index) => (
                    <button
                      key={`${result.metadata.document_id}-${result.metadata.page}-${index}`}
                      onClick={() => openSearchResult(result)}
                    >
                      <strong>{result.metadata.filename} · Page {result.metadata.page}</strong>
                      <span>{result.text}</span>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>
          <div className="top-actions">
            <button className="icon-button notification">
              <Icon name="bell" size={19} />
              <span />
            </button>
            <button className="primary-button" onClick={() => setShowUpload(true)}>
              <Icon name="upload" size={18} />
              Upload contract
            </button>
          </div>
        </header>

        {activeNav === "compare" ? (
          <CompareView
            documents={documents}
            firstDocumentId={firstDocumentId}
            secondDocumentId={secondDocumentId}
            onSelectFirst={setFirstDocumentId}
            onSelectSecond={setSecondDocumentId}
            onCompare={handleCompare}
            comparison={comparison}
            isComparing={isComparing}
            error={comparisonError}
          />
        ) : activeNav === "library" ? (
          <main className="page-content">
            <div className="page-heading">
              <div>
                <p className="eyebrow">KNOWLEDGE BASE</p>
                <h1>Clause library</h1>
                <p>Explore balanced contract language and understand common legal terms.</p>
              </div>
            </div>
            <div className="library-grid">
              {[
                "Payment terms",
                "Intellectual property",
                "Termination rights",
                "Liability & indemnity",
                "Confidentiality",
                "Non-compete clauses",
              ].map((x, i) => (
                <article key={x}>
                  <span className="library-number">0{i + 1}</span>
                  <div>
                    <h3>{x}</h3>
                    <p>
                      Learn what fair language looks like, what to watch for, and how to negotiate
                      it.
                    </p>
                    <button>
                      Open guide <Icon name="arrow" size={15} />
                    </button>
                  </div>
                </article>
              ))}
            </div>
          </main>
        ) : (
          <main className="page-content">
            <div className="page-heading">
              <div>
                <p className="eyebrow">
                  {activeNav === "documents" ? "CONTRACT WORKSPACE" : "ANALYSIS SUMMARY"}
                </p>
                <h1>{activeNav === "documents" ? "Your documents" : "Contract overview"}</h1>
                <p>
                  {activeNav === "documents"
                    ? `${documents.length} analyzed contract(s) available in your workspace.`
                    : analysis
                      ? `Risk review for ${analysis.filename}.`
                      : "Upload a contract to extract clauses and review risk."}
                </p>
              </div>
              <button className="outline-button">
                <Icon name="download" size={17} />
                Export report
              </button>
            </div>

            {activeNav === "documents" ? (
              <section className="documents-table">
                <div className="table-tools">
                  <div className="search small">
                    <Icon name="search" size={17} />
                    <input placeholder="Search contracts" />
                  </div>
                  <button className="filter-pill">
                    All documents <Icon name="chevron" size={14} />
                  </button>
                </div>
                {documents.map((document) => (
                  <button
                    key={document.document_id}
                    className="document-row"
                    onClick={() => {
                      setAnalysis(document);
                      setActiveNav("overview");
                      setActiveClause(document.clauses[0]?.id ?? "");
                    }}
                  >
                    <span className="file-tile">
                      <Icon name="file" size={20} />
                    </span>
                    <span className="doc-name">
                      <strong>{document.filename}</strong>
                      <small>{document.file_type} · {document.total_pages} pages</small>
                    </span>
                    <RiskBadge
                      level={
                        document.risk_counts.high > 0
                          ? "high"
                          : document.risk_counts.medium > 0
                            ? "medium"
                            : "low"
                      }
                    />
                    <span className="doc-date">{document.clauses.length} clauses</span>
                    <span className="doc-count">{document.total_characters.toLocaleString()} chars</span>
                    <Icon name="chevron" size={17} />
                  </button>
                ))}
                {documents.length === 0 && (
                  <p className="table-empty" role={documentsError ? "alert" : "status"}>
                    {documentsError ?? "No analyzed contracts yet. Upload one to get started."}
                  </p>
                )}
              </section>
            ) : analysis ? (
              <>
                <section className="contract-header">
                  <div className="file-tile">
                    <Icon name="file" />
                  </div>
                  <div className="contract-title">
                    <small>LATEST ANALYSIS</small>
                    <h2>{analysis.filename}</h2>
                    <p>
                      {analysis.file_type} · {analysis.total_pages} pages ·{" "}
                      {analysis.clauses.length} extracted clauses
                    </p>
                  </div>
                  <div className="score">
                    <div className="score-ring">
                      <strong>{analysis.overall_score}</strong>
                      <span>/ 100</span>
                    </div>
                    <div>
                      <RiskBadge level={overallRiskLevel} />
                      <small>
                        {analysis.risk_counts.high + analysis.risk_counts.medium} of{" "}
                        {analysis.clauses.length} clauses need attention
                      </small>
                    </div>
                  </div>
                </section>

                <section className="stats-grid">
                  <article>
                    <span className="stat-icon red">
                      <Icon name="alert" />
                    </span>
                    <div>
                      <small>HIGH RISK</small>
                      <strong>{analysis.risk_counts.high}</strong>
                      <p>Review before signing</p>
                    </div>
                  </article>
                  <article>
                    <span className="stat-icon amber">
                      <Icon name="clock" />
                    </span>
                    <div>
                      <small>MEDIUM RISK</small>
                      <strong>{analysis.risk_counts.medium}</strong>
                      <p>Worth negotiating</p>
                    </div>
                  </article>
                  <article>
                    <span className="stat-icon green">
                      <Icon name="shield" />
                    </span>
                    <div>
                      <small>LOW RISK</small>
                      <strong>{analysis.risk_counts.low}</strong>
                      <p>Looks balanced</p>
                    </div>
                  </article>
                  <article className="summary-stat">
                    <span className="stat-icon blue">
                      <Icon name="sparkles" />
                    </span>
                    <div>
                      <small>AI SUMMARY</small>
                      <p>
                        Found {analysis.clauses.length} clauses.{" "}
                        <b>{analysis.risk_counts.high} high-risk</b> and{" "}
                        <b>{analysis.risk_counts.medium} medium-risk</b> clause(s) need review.
                      </p>
                    </div>
                    <button
                      aria-label="Review highest-risk clause"
                      onClick={() =>
                        setActiveClause(
                          analysis.clauses.find((clause) => clause.risk_level === "high")?.id ??
                            analysis.clauses[0]?.id ??
                            "",
                        )
                      }
                    >
                      <Icon name="arrow" />
                    </button>
                  </article>
                </section>

                <section className="review-section">
                  <div className="section-head">
                    <div>
                      <h2>Clause review</h2>
                      <p>Select a clause to understand its impact and see a safer alternative.</p>
                    </div>
                    <div className="filter-tabs">
                      {RISK_FILTERS.map((x) => (
                        <button
                          key={x}
                          onClick={() => setFilter(x)}
                          className={filter === x ? "active" : ""}
                        >
                          {x[0].toUpperCase() + x.slice(1)}
                          {x !== "all" && (
                            <span>
                              {analysis.risk_counts[x]}
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="review-grid">
                    <div className="clause-list">
                      {visibleClauses.map((c) => (
                        <button
                          key={c.id}
                          className={`clause-item ${activeClause === c.id ? "selected" : ""}`}
                          onClick={() => setActiveClause(c.id)}
                        >
                          <span className={`risk-line ${c.risk_level}`} />
                          <span className="clause-copy">
                            <span>
                              <strong>{c.clause_type}</strong>
                              <small>Page {c.page_number}</small>
                            </span>
                            <p>{c.text}</p>
                            <RiskBadge level={c.risk_level} />
                          </span>
                          <Icon name="chevron" size={18} />
                        </button>
                      ))}
                    </div>
                    <aside className="insight-panel">
                      {current ? (
                        <>
                          <div className="insight-head">
                            <div>
                              <small>VERIFIED SOURCE QUOTE · PAGE {current.page_number}</small>
                              <h3>{current.clause_type}</h3>
                              <small>Risk score: {current.risk_score}/100</small>
                            </div>
                            <RiskBadge level={current.risk_level} />
                          </div>
                          <blockquote className="source-quote">{current.text}</blockquote>
                          <div className="key-terms">
                            <span>
                              <small>PARTY OBLIGATED</small>
                              <strong>{current.party_obligated}</strong>
                            </span>
                            <span>
                              <small>DEADLINES</small>
                              <strong>{current.deadlines.join(", ") || "None identified"}</strong>
                            </span>
                            <span>
                              <small>MONEY AMOUNTS</small>
                              <strong>{current.money_amounts.join(", ") || "None identified"}</strong>
                            </span>
                          </div>
                          <div className="explanation">
                            <span className={`explain-icon ${current.risk_level}`}>
                              <Icon name="sparkles" size={18} />
                            </span>
                            <div>
                              <h4>What this means for you</h4>
                              <p>{current.explanation}</p>
                              <p>
                                <b>Difference from the reference:</b> {current.deviation}
                              </p>
                            </div>
                          </div>
                          <div className="reference">
                            <div>
                              <Icon name="book" size={16} />
                              <strong>
                                {current.standard_reference?.title ??
                                  "No matching standard clause found"}
                              </strong>
                            </div>
                            {current.standard_reference && (
                              <p>{current.standard_reference.text}</p>
                            )}
                            <small>
                              Retrieved from the local standard clause library using ChromaDB.
                            </small>
                          </div>
                          <div className="rewrite">
                            <div className="rewrite-head">
                              <span>
                                <Icon name="check" size={16} />
                                SUGGESTED SAFER WORDING
                              </span>
                              <button onClick={() => void copyRewrite(current.suggested_rewrite)}>
                                Copy
                              </button>
                            </div>
                            <p>“{current.suggested_rewrite}”</p>
                            {copyError && <small role="alert">{copyError}</small>}
                          </div>
                        </>
                      ) : (
                        <p>Select a clause to review its risk and reference wording.</p>
                      )}
                    </aside>
                  </div>
                </section>
                <p className="legal-disclaimer">{analysis.disclaimer}</p>
              </>
            ) : (
              <section className="empty-state">
                <span className="feature-icon purple">
                  <Icon name="file" />
                </span>
                <h2>Start with a contract</h2>
                <p>
                  Upload a PDF or image to extract clauses, compare them with the standard
                  clause library, and review cited risk explanations and suggested edits.
                </p>
                {documentsError && <p role="alert">{documentsError}</p>}
                <button className="primary-button" onClick={() => setShowUpload(true)}>
                  <Icon name="upload" size={18} />
                  Upload contract
                </button>
                <small>AI-generated review is informational, not legal advice.</small>
              </section>
            )}
          </main>
        )}
      </div>
      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onUpload={handleUpload}
          isUploading={isUploading}
          uploadError={uploadError}
          uploadSuccess={uploadSuccess}
        />
      )}
    </div>
  );
}

export default App;
