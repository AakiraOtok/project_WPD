# WEAPON DETECTION

### Đôi lời

Đầu tiên em xin được dành lời cảm ơn đến cô TS. Dương Việt Hằng - cô chủ nhiệm và là giáo viên hướng dẫn của em trong đề tài lần này - đã hỗ trợ em rất nhiều
trong quá trình tìm tòi và học hỏi. Và em cũng không quên gửi lời cảm ơn tới các bạn/anh chị (anh Nhân, anh Hào, bạn Trâm và các bạn trong sever Discord TIU) 
đã giúp em không những trong việc tìm hiểu CV mà còn những kiến thức trên lớp, điều đó đã tạo điều kiện cho em có thêm thời gian để code và nghiên cứu CV.

Đây sẽ là repo mà em lưu lại code của mình trong đề tài lần này, em sẽ liên tục cập nhật trong lúc code. Đây là lần đầu tiên em thực dùng github phục vụ cho việc học tập của mình, 
rất mong nhận được sự góp ý và lời khuyên từ các thầy, cô, anh chị cũng như các bạn đang xem repo.

Mọi ý kiến đóng góp hay thắc mắc xin gửi về địa chỉ email : 22520847@gm.uit.edu.vn

### Chú ý

Code đang liên tục được cập nhật, có thể trong quá trình cập nhật thì code không còn tương thích với những setting em đã đề cập ở phía dưới vào một thời điểm nào đó, em sẽ cố gắng
update note này thường xuyên nhất có thể

### Môi trường

```python = 3.10.11```

```cuda   = 11.8```


### pakage
```pytorch = 2.0.1```

```cv2 = 4.7.0```

```tqdm = 4.65.0```

```numpy = 1.24.3```

```matplotlib = 3.7.1```

Để cho đơn giản thì mọi người cứ tải phiên bản mới nhất của các pagkage trên là được.

### Về dataset

Em đang sử dụng VOC dataset, cụ thể và đường link tải nằm ở ngay dưới đây :
- VOC2012 train/val : http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
- VOC2007 train/val : http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
- VOC2007 test      : http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

### live cam
Mọi người đã có thể clone về và run file live.py (sử dụng camera của máy),  lưu ý chỉ hỗ trợ máy có gpu nvdia, các bước cần thực hiện như sau :
- Cài đặt môi trường và các package như đã nêu ở trên
- Tải về checkpoint tại : https://drive.google.com/file/d/1zBVFvuIhpvnVwGYmu82osUu71ugGF_zb/view?usp=sharing
- Ở file live.py, sửa pretrain_path lại thành đường dẫn checkpoint vừa tải về

Sau khi thực hiện các bước ở trên, bạn đã có thể test thử model rồi.
