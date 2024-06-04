// Developed by Mena S.A. Kamel (mena.sa.kamel@gmail.com | mena.kamel@sanofi.com)
// date: June 4, 2024
// Requirements: Set environment variables for AWS credentials:
//     - AWS_ACCESS_KEY: AWS key ID
//     - AWS_SECRET_KEY: AWS secret access key

/**
 * @file Annotation_S3_Upload.js
 */

/**
 * @namespace Annotation_S3_Upload
 * @classdesc The root namespace for Annotation_S3_Upload.
 */
var Annotation_S3_Upload;
Annotation_S3_Upload = {
  name: "Annotation_S3_Upload Plugin",
};

/**
 * This method is called when the document is loaded.
 * The container element is a div where the plugin options will be displayed.
 * @summary After setting up the tmapp object, initialize it*/
Annotation_S3_Upload.init = function (container) {
  container.innerHTML = "";
  const JSZipScript = document.createElement("script");
  JSZipScript.src =
    "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.6.0/jszip.min.js";
  document.head.appendChild(JSZipScript);
  bucketNameField = Annotation_S3_Upload.s3BucketnameField(container);
  s3KeyField = Annotation_S3_Upload.s3LocationField(container);
  Annotation_S3_Upload.UploadButton(container, bucketNameField, s3KeyField);
};

Annotation_S3_Upload.getCurrentPath = function () {
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  return urlParams.get("path");
};

Annotation_S3_Upload.s3BucketnameField =
  function s3Bucketname(container) {
    const bucketNameRow = HTMLElementUtils.createRow({});
    const bucketNameDescriptionCol = HTMLElementUtils.createColumn({
      width: 6,
      extraAttributes: { class: "d-flex justify-content-start" },
    });
    const bucketNameInputCol = HTMLElementUtils.createColumn({
      width: 6,
      extraAttributes: { class: "d-flex justify-content-end" },
    });

    const bucketNameDescription = document.createElement('div');
    bucketNameDescription.innerText = "S3 bucket name";
    const bucketNameInput = HTMLElementUtils.inputTypeText({
        extraAttributes: {
            size: 100,
            placeholder: "class",
            value: "sample-bucket-name",
            class: "col input-sm form-control form-control-sm",
            style: "width: 300px;",
        }
    });
    container.appendChild(bucketNameRow);
    bucketNameRow.appendChild(bucketNameDescriptionCol);
    bucketNameRow.appendChild(bucketNameInputCol);
    bucketNameDescriptionCol.appendChild(bucketNameDescription);
    bucketNameInputCol.appendChild(bucketNameInput);
    return bucketNameInput;
  };

Annotation_S3_Upload.s3LocationField =
  function s3Location(container) {
    const s3KeyRow = HTMLElementUtils.createRow({});
    const s3KeyDescriptionCol = HTMLElementUtils.createColumn({
      width: 6,
      extraAttributes: { class: "d-flex justify-content-start" },
    });
    const s3KeyInputCol = HTMLElementUtils.createColumn({
      width: 6,
      extraAttributes: { class: "d-flex justify-content-end" },
    });

    const s3KeyDescription = document.createElement('div');
    s3KeyDescription.innerText = "S3 file location";
    const s3KeyInput = HTMLElementUtils.inputTypeText({
        extraAttributes: {
            size: 100,
            placeholder: "class",
            value: "path/to/my-object.geojson",
            class: "col input-sm form-control form-control-sm",
            style: "width: 300px;",
        }
    });
    container.appendChild(s3KeyRow);
    s3KeyRow.appendChild(s3KeyDescriptionCol);
    s3KeyRow.appendChild(s3KeyInputCol);
    s3KeyDescriptionCol.appendChild(s3KeyDescription);
    s3KeyInputCol.appendChild(s3KeyInput);
    return s3KeyInput;
  };

Annotation_S3_Upload.UploadButton =
  function UploadButton(container, bucketNameField, s3KeyField) {
    const UploadButtonRow = HTMLElementUtils.createRow({});
    const UploadButton = HTMLElementUtils.createButton({
      extraAttributes: { class: "btn btn-primary" },
    });
    UploadButton.innerHTML = "Upload annotations to S3";
    UploadButton.addEventListener("click", async (event) => {
      UploadButton.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
      <span class="sr-only">Wait...</span>`;
      UploadButton.setAttribute("disabled", "true");
      try {
        geojsons = regionUtils.regions2GeoJSON(regionUtils._regions);
        console.log(geojsons);
        console.log(bucketNameField.value);
        console.log(s3KeyField.value);
        await Annotation_S3_Upload.uploadAnnotations(geojsons, bucketNameField.value, s3KeyField.value);
      } 
      catch {
        interfaceUtils.alert("Error Uploading Annotations!");
      }
      UploadButton.innerHTML = "Upload annotations to S3";
      UploadButton.removeAttribute("disabled");
    });
    container.appendChild(UploadButtonRow);
    UploadButtonRow.appendChild(UploadButton);
  };

Annotation_S3_Upload.uploadAnnotations = async function (geojsons, bucketName, fileLocation) {
  // Get current path
  const response = await fetch(
    `/plugins/Annotation_S3_Upload/upload_annotations`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        path: Annotation_S3_Upload.getCurrentPath(),
        bucket_name: bucketName,
        file_location: fileLocation,
        geojsons: geojsons,
      }),
    }
  );
  return response.json();
};