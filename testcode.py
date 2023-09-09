import zipfile

file_to_zip = "H:/checkpoint/detections_test-dev2017_bb.json"
zipfile_path = "H:/checkpoint/detections_test-dev2017_bb.zip"

# Tạo một đối tượng ZipFile
with zipfile.ZipFile(zipfile_path, 'w') as zipf:
    # Thêm tệp tin vào tệp tin ZIP
    zipf.write(file_to_zip, "detections_test-dev2017_bb.json")

print(f"Tệp tin '{file_to_zip}' đã được nén thành '{zipfile_path}' thành công.")

