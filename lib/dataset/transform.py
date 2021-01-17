from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image


def read_xray_image(path, voi_lut=True, fix_monochrome=True):
    """
    Read dicom image and convert numpy array with pixel value in range of 0 to 255
    Args:
        path: path to dicom file
        voi_lut: don't know what the fuck is this
        fix_monochrome: convert to Photometric Interpretation type MONOCHROME2
    Returns:
        Image array (numpy array) of xray photo with pixel value in range of [0,255]
    """
    # Read file
    dicom_data = dcmread(path)
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom_data.pixel_array, dicom_data)
    else:
        data = dicom_data.pixel_array
    # depending on this value, X-ray may look inverted - fix that
    if fix_monochrome and dicom_data.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    patient_sex = dicom_data.PatientSex

    return data, patient_sex


def save_array_to_image(pixel_array, name):
    """
    Convert image arrray to image file 
    """
    img = Image.fromarray(pixel_array, 'L')
    img.save(name + '.jpg')