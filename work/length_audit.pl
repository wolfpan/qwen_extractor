#!/usr/bin/perl
use strict; use warnings; use JSON::PP;
binmode(STDOUT, ':utf8');
for my $f (@ARGV) {
  open my $h, '<:raw', $f or do { warn "skip $f: $!"; next };
  local $/;
  my $raw = <$h>; close $h;
  my $d = eval { JSON::PP->new->decode($raw) };
  unless ($d) { warn "parse fail $f: $@"; next }
  my (@lens, $over200, $over180, $n, $skipn);
  for my $s (@$d) {
    my $o = eval { JSON::PP->new->decode($s->{output}) };
    if (!$o) { next }
    my $sum = $o->{"\x{6838}\x{5fc3}\x{6458}\x{8981}"} // '';   # 核心摘要
    if ($sum eq 'SKIP') { $skipn++; next }
    $n++; my $L = length $sum; push @lens, $L;
    $over200++ if $L > 200;
    $over180++ if $L > 180;
  }
  @lens = sort { $a <=> $b } @lens;
  my $p = sub { my $q = shift; $lens[int($#lens * $q)] // 0 };
  printf "%s: nonSKIP=%d SKIP=%d | len min=%d p50=%d p90=%d max=%d | >180: %d | >200: %d\n",
    $f, $n, $skipn // 0, $lens[0] // 0, $p->(0.5), $p->(0.9), $lens[-1] // 0, $over180 // 0, $over200 // 0;
}
