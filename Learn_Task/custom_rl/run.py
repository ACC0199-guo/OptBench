# main.py
import subprocess
import sys


def run_script(script_name):
    print(f"正在执行 {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ {script_name} 执行失败，退出码：{result.returncode}")
        if result.stderr:
            print("错误信息：", result.stderr)
        exit(result.returncode)
    else:
        print(f"✅ {script_name} 执行成功")


if __name__ == "__main__":
    scripts = ["custom_sac.py", "custom_mpc_sac.py", "custom_momentum_sac.py"]  # 替换为你的脚本名

    for script in scripts:
        run_script(script)

    print("🎉 所有脚本已顺序执行完毕！")
