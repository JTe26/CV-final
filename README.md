**基于 TensoRF 与 3D Gaussian Splatting 的物体重建与视图合成**


TensoRF 实验：
- **TensoRF下载**：
  将TensoRF相关文件下载下来。
  ```bash
  git clone https://github.com/apchenstu/TensoRF
  ```
  
- **数据预处理**：  
  可以使用 COLMAP 工具恢复每张图像对应的相机姿态信息：

  1. **特征提取**  
     ```bash
     colmap feature_extractor --database_path database.db --image_path my_project
     ```

  2. **顺序匹配**  
     ```bash
     colmap sequential_matcher --database_path database.db
     ```

  3. **稀疏重建**  
     构建稀疏点云，并估计每张图像的内外参，输出在 `sparse/0`。  
     ```bash
     colmap mapper --database_path database.db --image_path my_project --output_path sparse
     ```
     注意该步骤会根据图像数量耗费不同的时间，如果图像在300个以上可能会耗费一小时以上。

  4. **转换为 TXT 格式**   
     ```bash
     colmap model_converter --input_path sparse/0 --output_path sparse/0 --output_type TXT
     ```


  5. **转换为 NeRF 格式**   
     ```bash
     python colmap2nerf.py --colmap_dir sparse/0 --images my_project --out_dir ./my_project
     ```

  6. **tranforms.json划分数据集（train/val/test）**
     ```bash
     python split.py
     ```

- **模型训练**：
  运行下面命令就可以开始训练，300张图片训练时间大约在45分钟左右。
  ```bash
  python train.py --config configs/my_own_data.txt
  ```

- **视频渲染**:
  ```bash
  python train.py --config configs/my_own_data.txt --ckpt path/to/your/checkpoint --render_only 1 --render_test 1 
  ```
***

3D Gaussian Splatting 实验：
- **Repo预备**：
  ```bash
  git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
  pip install plyfile tqdm
  pip install submodules/diff-gaussian-rasterization
  pip install submodules/simple-knn
  ```

- **数据预处理**：
  在TensoRF中需要好几步的预处理在这里只需要一步即可：
  ```bash
  python convert.py -s $FOLDER_PATH
  ```

- **模型训练**：
  ```bash
  python train.py -s $FOLDER_PATH -m $FOLDER_PATH/output -w
  ```
  对于这个模型进行可视化：
  ```bash
  SIBR_gaussianViewer_app -m $FOLDER_PATH/output
  ```

  


