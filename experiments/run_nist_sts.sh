#!/bin/sh
# Official NIST STS 2.1.2 on XC-PDA ciphertext (all 15 tests, 1,000 sequences of 10^6 bits).
# Usage, from the repository root:  sh experiments/run_nist_sts.sh audio   (or: silence)
set -e
MODE=${1:-audio}
WORK=${WORK:-nist_work}; mkdir -p "$WORK"; cd "$WORK"
if [ ! -d sts ]; then
  git clone --depth 1 https://github.com/terrillmoore/NIST-Statistical-Test-Suite src_sts
  cp -r src_sts/sts sts
fi
cd sts && mkdir -p obj && make -s && cd experiments && for d in AlgorithmTesting BBS CCG G-SHA1 LCG MODEXP MS QCG1 QCG2 XOR; do mkdir -p $d; done && sh create-dir-script >/dev/null 2>&1 || true; cd ..
python3 ../../experiments/make_sts_input.py "$MODE" "../$MODE.bin"
printf '0\n../%s.bin\n1\n0\n1000\n1\n' "$MODE" | ./assess 1000000 > /dev/null || true
cp experiments/AlgorithmTesting/finalAnalysisReport.txt "../../results/nist_sts/finalAnalysisReport_$MODE.txt"
echo "report: results/nist_sts/finalAnalysisReport_$MODE.txt"
