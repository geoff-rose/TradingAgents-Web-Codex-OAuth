# Sourced ticker memory

The announcement feed's **Facts** button opens the evidence stored for a ticker.
It shows each claim, its supporting quotation and source page, document-processing
progress, and the latest comparison. The authenticated API is
`GET /api/asx/memory/{ticker}` (for example `SRL` or `SRL.AX`).

The existing signals SQLite database now also contains `memory_documents`,
`memory_sections`, `memory_facts` and `memory_comparisons`. Facts and comparisons
are append-only. Conflicting and superseded claims retain their original source
and publication time. Model/extraction version and exact document text hashes
separate revisions. Identical sections for the same ticker/model reuse extraction.

PDF extraction retains all pages with page markers. Text sections up to 24,000
characters are processed independently, with overlap when a page must be split.
An image-only page that cannot be read blocks scoring and requests OCR/manual
review through the existing document-failure status. OCR and interpretation of
complex diagrams are not implemented. Text extraction is not a guarantee that
every graphical detail or material fact is understood.

The extraction model returns material claims with verbatim quotations. Unsupported
quotes, malformed responses and incomplete section processing cannot produce a
new score. Every extracted fact must be represented in the comparison. Current
and earlier fact IDs are validated. Unmatched claims are marked uncertain, not
proven new. Conditional versus binding funding and meaningful milestone changes
are assessed explicitly; previous disclosure is not assumed fully priced in.

History builds lazily. Before scoring, up to two recent relevant documents within
365 days are indexed, oldest first. Candidate titles only prioritise retrieval;
facts and scores come from documents. Missing older coverage is exposed in each
comparison. This is a bounded starting history, not a complete company archive.
Further history accumulates as announcements are processed. Retrieval is limited
to 80 relevant prior facts, prioritised by topic and overlapping subject words;
when more match, the comparison records that retrieval was limited.

Each classifier refresh permits at most 16 additional extraction calls. Long
documents resume on later refreshes using cached completed sections. Existing
document-backed scores remain visible until reassessment finishes; replacements
are archived by the existing signal-history mechanism. Production uses prompt
`v7-ticker-memory`. Experimental runs reuse extraction but keep their own scores.

Earlier facts are filtered strictly before the current publication timestamp,
including timezone normalisation. Backfilled facts may have later extraction
timestamps; they provide historical document evidence, not live score outcomes.
The historical score and prospective-return tables retain their existing separation.
