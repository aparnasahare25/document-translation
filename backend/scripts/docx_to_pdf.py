import os
from dotenv import load_dotenv
import fitz  
load_dotenv(override=True)

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.create_pdf_job import CreatePDFJob
from adobe.pdfservices.operation.pdfjobs.result.create_pdf_result import CreatePDFResult


class CreatePDFFromDOCX:
    def __init__(self, input_docx_path: str, output_pdf_path: str):
        try:
            file = open(input_docx_path, 'rb')
            input_stream = file.read()
            file.close()

            # Initial setup, create credentials instance
            credentials = ServicePrincipalCredentials(
                client_id=os.getenv('PDF_SERVICES_CLIENT_ID'),
                client_secret=os.getenv('PDF_SERVICES_CLIENT_SECRET')
            )

            # Creates a PDF Services instance
            pdf_services = PDFServices(credentials=credentials)

            # Creates an asset(s) from source file(s) and upload
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.DOCX)

            # Creates a new job instance
            create_pdf_job = CreatePDFJob(input_asset)

            # Submit the job and gets the job result
            location = pdf_services.submit(create_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, CreatePDFResult)

            # Get content from the resulting asset(s)
            result_asset: CloudAsset = pdf_services_response.get_result().get_asset()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            # Creates an output stream and copy stream asset's content to it
            output_dir = os.path.dirname(output_pdf_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_pdf_path, "wb") as file:
                file.write(stream_asset.get_input_stream())

            # Remove bookmarks added by Adobe during conversion
            doc = fitz.open(output_pdf_path)
            doc.set_toc([])  # clears all bookmarks
            tmp_path = output_pdf_path + ".tmp"
            doc.save(tmp_path)
            doc.close()
            os.replace(tmp_path, output_pdf_path)  # atomically replace original
            print(f"PDF saved without bookmarks: {output_pdf_path}")

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            print(f'Exception encountered while executing operation: {e}')


if __name__ == "__main__":
    input_docx_path = r"C:\gen_ai\document-translation\backend\output.docx"
    output_pdf_path = r"output.pdf"
    CreatePDFFromDOCX(input_docx_path, output_pdf_path)
