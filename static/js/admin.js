
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("document");
    const uploadButton = document.getElementById("uploadButton");
    const loadingOverlay = document.getElementById("ingestLoading");

    if (!form || !fileInput || !uploadButton || !loadingOverlay) {
        return;
    }

    form.addEventListener("submit", (event) => {
        const file = fileInput.files?.[0];

        if (!file) {
            event.preventDefault();

            if (typeof Swal !== "undefined") {
                Swal.fire({
                    icon: "warning",
                    title: "Dokumen belum dipilih",
                    text: "Pilih file PDF terlebih dahulu."
                });
            } else {
                alert("Pilih file PDF terlebih dahulu.");
            }

            return;
        }

        const isPdf =
            file.type === "application/pdf" ||
            file.name.toLowerCase().endsWith(".pdf");

        if (!isPdf) {
            event.preventDefault();

            if (typeof Swal !== "undefined") {
                Swal.fire({
                    icon: "error",
                    title: "Format tidak valid",
                    text: "Silakan pilih dokumen PDF."
                });
            } else {
                alert("Silakan pilih dokumen PDF.");
            }

            return;
        }

        // Biarkan browser mengirim form secara normal ke Flask.
        uploadButton.disabled = true;
        uploadButton.querySelector("span").textContent =
            "Sedang mengunggah dan memproses...";

        loadingOverlay.classList.add("active");
    });
});
