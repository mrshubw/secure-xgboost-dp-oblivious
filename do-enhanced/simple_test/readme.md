目录结构：
- data： 存放模型文件于数据集
- data/higgs/modeld8n5.model: 适用higgs数据集的模型文件，深度为8，树的数量为5。经过了key.txt加密。
- data/higgs/data1000.txt: 1000条higgs数据，明文，libsvm数据格式。
- data/higgs/data1000.enc：data1000.txt经过key.txt加密后的文件。
- key.txt：加密密钥文件。
- predict.py：执行脚本，进行测试。

测试步骤：

- 将文件解压后放在secure xgboost项目下，根据predict.PY文件相对于secure xgboost的路径，修改predict.py中的HOME_DIR。

- 找到secure xgboost项目的/enclave/include/enclave_context.h文件，修改其中的generate_symm_key函数为一下内容：

    ```
    bool generate_symm_key() {
        // generate_random(m_symm_key, CIPHER_KEY_SIZE);
        
        // In the DP Oblivious experiment, in order to separate training from inference, the key is forced to be 0. 
        // Notice, this is not secure.
        memset(m_symm_key, 0, CIPHER_KEY_SIZE);
        }
    ```
    这是因为原始的generate_symm_key函数在每一次运行时生成的密钥都不一样，导致训练的模型在推理时无法解密。因此这里固定密钥为0，方便预先训练文件进行推断。

- 运行predict.py.