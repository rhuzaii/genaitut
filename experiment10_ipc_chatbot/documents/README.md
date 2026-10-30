# Documents

## The Indian Penal Code

The IPC is not committed to this repository. Download it yourself and save it
here as `ipc.pdf`.

Official source, India Code, the Government of India legislative repository:

    https://www.indiacode.nic.in

Search for "Indian Penal Code, 1860" and download the PDF of the bare Act.

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
