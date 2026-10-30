# Documents

## The Indian Penal Code

`ipc.pdf` is committed here: 119 pages, 1.1 MB, the bare Act as published by
India Code. It is a Government of India public document.

Do not use the `.txt` bitstream that India Code offers alongside the PDF. It
is truncated to about a fifth of the Act, stopping partway through the private
defence chapter, and it contains no operative text for section 302 or anything
after it. The truncation is silent: the file looks complete and parses without
error. Use the PDF.

Official source, India Code, the Government of India legislative repository.
The site migrated from indiacode.nic.in to indiacode.gov.in, and the IPC is
filed under its repealed title because the Bharatiya Nyaya Sanhita replaced
it, so searching the Acts collection does not find it:

    https://indiacode.gov.in/items/972afbe0-a2de-415d-9df5-8d70038056ad

Direct PDF:

    https://indiacode.gov.in/server/api/core/bitstreams/337c7a03-60c6-4953-8dd6-90e65c343629/content

Then build the index:

    python build_index.py --document documents/ipc.pdf

Note that the Bharatiya Nyaya Sanhita, 2023 has replaced the IPC for offences
committed on or after 1 July 2024. The lab exercise names the IPC, so use the
IPC, but say which document you indexed in your report, since the chatbot can
only ever be as current as the text it was given.

## sample_statute.txt

A short fictional statute, written for this project, used to test the parser,
the chunker and the retrieval pipeline without needing the real document.

It is **not law**. It exists only so the code can be run and verified end to
end before the IPC is downloaded. Do not cite it, and do not use it in your
report as though it were a legal source.
