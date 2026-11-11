#!/usr/bin/perl
use strict; use warnings; use utf8; use Encode qw(decode_utf8);
binmode(STDOUT, ':utf8');
my (@lens, $n, $skip, $o180, $o200);
while (my $line = <>) {
  next unless $line =~ /^\s*"output":\s*"((?:[^"\\]|\\.)*)"/;
  my $out = $1;
  $out =~ s/\\(.)/$1 eq 'n' ? "\n" : $1/ge;          # 外层反转义
  $out = decode_utf8($out);
  next unless $out =~ /"核心摘要":"((?:[^"\\]|\\.)*)"/;
  my $sum = $1;
  $sum =~ s/\\(.)/$1 eq 'n' ? "\n" : $1/ge;
  if ($sum eq 'SKIP') { $skip++; next }
  $n++; my $L = length $sum; push @lens, $L;
  $o180++ if $L > 180; $o200++ if $L > 200;
}
@lens = sort { $a <=> $b } @lens;
exit unless $n;
printf "nonSKIP=%d SKIP=%d | min=%d p50=%d p90=%d max=%d | >180chars=%d | >200chars=%d\n",
  $n, $skip // 0, $lens[0], $lens[int($#lens*0.5)], $lens[int($#lens*0.9)], $lens[-1], $o180 // 0, $o200 // 0;
