import qrcode
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_M,
    box_size=10,
    border=4,
)
qr.add_data('<?xml version="1.0" encoding="UTF-8"?>\n<PrintLetterBarcodeData uid="743650975966" name="Khushbu Rani Sahu" gender="F" yob="1997" co="D/O: Mohan Lal Sahu" house="House No.73" street="Ward 05" lm="Haldi" loc="Haldi" vtc="Haldi" po="Hardi" dist="Balod" subdist="Gunderdehi" state="Chhattisgarh" pc="491222" dob="05/12/1997"/>')
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("6f4.png")