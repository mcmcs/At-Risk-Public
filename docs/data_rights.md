# Data rights and access

Reviewed 12 November 2026. The public data-free edition distributes the authors' code and
reported research outputs, but no raw empirical observations, downloaded
archives, or saved empirical forecast panels. The MIT software license does
not license provider data.

## Why the empirical data are omitted

The [FRED-MD page](https://www.stlouisfed.org/research/economists/mccracken/fred-databases)
offers historical databases for empirical research and replication. Public
availability is not a blanket redistribution license for every component.
The [St. Louis Fed website terms](https://www.stlouisfed.org/about-us/legal-information)
permit research and informational use but reserve third-party copyright
permissions to the owners. The [FRED service terms](https://fred.stlouisfed.org/legal/terms/)
also preserve the restrictions of individual data owners. For example, the
[University of Michigan consumer sentiment series](https://fred.stlouisfed.org/series/UMCSENT)
carries a third-party copyright notice and attribution instructions.

We have not established redistribution permission for every series in the
historical panels. Consequently, this submission omits the empirical inputs
and provides provider downloads and exact checks instead. This is a packaging
decision, not a claim that every FRED-MD series is legally non-shareable.
Users must follow the providers' current terms and any series-specific
conditions; the download helper does not confer additional rights.

## Obtaining access

The required FRED-MD ZIP archives, New York Fed workbook, and FRED USRECM CSV
are offered through public download links. The documented download route uses
no login, API key, application, or payment. Allow several minutes for roughly
80 MB of archives, depending on the connection. There is no provider approval
process built into this package. Any permission needed for additional uses
must be obtained separately from the relevant rights holder.

Use [data_downloads.md](data_downloads.md) for exact vintages, sample bounds,
installation commands, manual placement, and checks. Public availability can
change. If a provider removes or alters a required file, stop and contact
Minchul Shin at minchul.shin@phil.frb.org. Do not replace historical vintages
with current data or alter a checksum. The authors cannot promise temporary
redistribution of provider data without the necessary permission.

## Journal reproducibility review

This note documents the data omission and provider acquisition route.
The journal's reproducibility check remains pending.

Empirical intermediate panels are also omitted. They are recreated by the
full workflow; [intermediate_data.md](intermediate_data.md) identifies their
producers. The included reference tables and figures allow comparison but
cannot replace those panels for regeneration. The literature-survey CSVs
are author-compiled documentary material and remain included.
