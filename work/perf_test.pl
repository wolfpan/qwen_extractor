#!/usr/bin/perl
# 快速性能测试：从 v4 验证集取一条真实样本，按生产参数请求 Ollama /api/chat，
# 解析 eval_count/eval_duration 等计时字段，输出 prefill 与生成速度。
# 用法: perl work/perf_test.pl [model] [样本序号|median]   （默认 qwen-extractor:0.8b median）
use strict; use warnings; use utf8; use Encode qw(decode_utf8);
binmode(STDOUT, ':utf8');

my $model = $ARGV[0] // 'qwen-extractor:0.8b';
my $pick_arg = $ARGV[1] // 'median';
my $host = $ENV{OLLAMA_HOST} // 'http://10.8.0.5:11434';

# 逐行提取 instruction/input（值本身已是 JSON 转义串，可直接嵌入请求，不做整体 decode）
open my $fh, '<:raw', 'data/dataset_v4_val.json' or die "open val: $!";
my (@instr, @input);
my $cur_i = '';
while (my $line = <$fh>) {
    if ($line =~ /^\s*"instruction":\s*"((?:[^"\\]|\\.)*)"/) { $cur_i = $1 }
    elsif ($line =~ /^\s*"input":\s*"((?:[^"\\]|\\.)*)"/) {
        push @instr, $cur_i; push @input, $1;
    }
}
close $fh;
die "no samples\n" unless @input;

# 过滤空/超短正文（SKIP 类样本），按原始长度取中位或指定序号
my @ok = grep { length($input[$_]) > 200 } 0..$#input;
die "no non-SKIP samples\n" unless @ok;
my $pick;
if ($pick_arg eq 'median' || $pick_arg eq 'shortest' || $pick_arg eq 'longest') {
    my @by_len = sort { length($input[$a]) <=> length($input[$b]) } @ok;
    $pick = $pick_arg eq 'shortest' ? $by_len[0]
          : $pick_arg eq 'longest'  ? $by_len[-1]
          : $by_len[int($#by_len / 2)];
} else {
    $pick = int($pick_arg);
    die "bad index $pick\n" unless $pick >= 0 && $pick < @input;
}

my $req = '{"model":"' . $model . '","stream":false,"think":false,"format":"json",'
    . '"messages":[{"role":"system","content":"' . $instr[$pick] . '"},'
    . '{"role":"user","content":"' . $input[$pick] . '"}],'
    . '"options":{"temperature":0.2,"num_ctx":4096,"num_predict":500,"repeat_penalty":1.05}}';
open my $out, '>:raw', 'work/perf_req.json' or die $!;
print $out $req; close $out;

my $rc = system(qq{curl -s -m 300 -d \@work/perf_req.json -H "Content-Type: application/json" $host/api/chat -o work/perf_resp.json});
die "curl failed rc=$rc\n" if $rc;

open my $rf, '<:raw', 'work/perf_resp.json' or die $!;
my $resp = do { local $/; <$rf> }; close $rf;
if ($resp =~ /"error"\s*:\s*"((?:[^"\\]|\\.)*)"/) { die "ollama error: $1\n" }

my %f;
for my $k (qw(total_duration load_duration prompt_eval_count prompt_eval_duration eval_count eval_duration)) {
    ($f{$k}) = $resp =~ /"$k"\s*:\s*(\d+)/;
}
my ($content) = $resp =~ /"content"\s*:\s*"((?:[^"\\]|\\.)*)"/;
$content //= '';
$content =~ s/\\n/\n/g; $content =~ s/\\"/"/g; $content =~ s/\\\\/\\/g;

my $gen_tps  = $f{eval_duration}       ? $f{eval_count}       / ($f{eval_duration}       / 1e9) : 0;
my $pre_tps  = $f{prompt_eval_duration} ? $f{prompt_eval_count} / ($f{prompt_eval_duration} / 1e9) : 0;
printf "model=%s sample=val#%d input_chars=%d\n", $model, $pick, length(decode_utf8($input[$pick]));
printf "prompt_tokens=%s prefill=%.1f tok/s | gen_tokens=%s speed=%.1f tok/s\n",
    $f{prompt_eval_count} // '?', $pre_tps, $f{eval_count} // '?', $gen_tps;
my ($done) = $resp =~ /"done_reason"\s*:\s*"([^"]*)"/;
printf "load=%.2fs total=%.2fs done_reason=%s\n", ($f{load_duration} // 0) / 1e9, ($f{total_duration} // 0) / 1e9, $done // '?';
print "output: ", decode_utf8($content), "\n";
