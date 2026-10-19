document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("uploadForm");
    const uploadButton = document.getElementById("uploadButton");
    const documentInput = document.getElementById("document");
    const ingestLoading = document.getElementById("ingestLoading");

    if (!uploadForm) return;

    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        if (!documentInput.files.length) {
            Swal.fire({
                icon: "warning",
                title: "File belum dipilih",
                text: "Silakan pilih file PDF terlebih dahulu.",
                confirmButtonColor: "#6544d8"
            });
            return;
        }

        const file = documentInput.files[0];

        if (!file.name.toLowerCase().endsWith(".pdf")) {
            Swal.fire({
                icon: "error",
                title: "Format file tidak valid",
                text: "File yang diperbolehkan hanya PDF.",
                confirmButtonColor: "#6544d8"
            });
            return;
        }

        const formData = new FormData(uploadForm);

        // Tampilkan loading
        uploadForm.classList.add("loading");
        uploadButton.disabled = true;
        uploadButton.textContent = "Sedang memproses...";
        ingestLoading.classList.add("active");

        try {
            const response = await fetch(uploadForm.action, {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(
                    data.message || "Gagal melakukan ingest dokumen."
                );
            }

            // Sembunyikan loading terlebih dahulu
            ingestLoading.classList.remove("active");

            await Swal.fire({
                icon: "success",
                title: "Upload Berhasil",
                text: data.message,
                confirmButtonColor: "#6544d8"
            });

            documentInput.value = "";

        } catch (error) {
            console.error("Upload error:", error);

            // Sembunyikan loading
            ingestLoading.classList.remove("active");

            Swal.fire({
                icon: "error",
                title: "Upload Gagal",
                text: error.message || "Terjadi kesalahan saat memproses dokumen.",
                confirmButtonColor: "#6544d8"
            });

        } finally {
            uploadForm.classList.remove("loading");
            uploadButton.disabled = false;
            uploadButton.textContent = "Upload & Ingest ke Knowledge Base";
        }
    });
});