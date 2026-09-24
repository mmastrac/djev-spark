#!/usr/bin/env bash
# Track the vLLM perf branches in vllm-stack.tsv against the fork and upstream.
#
#   scripts/vllm-stack.sh status
#       One line per branch: whether the fork branch moved past the pinned
#       head, the upstream PR's state, and whether the image's base nightly
#       already contains the merged PR. Needs gh.
#
#   scripts/vllm-stack.sh build <nightly commit> [workdir]
#       Build a new stack on a nightly's commit: each branch's own diff
#       (against its merge base with upstream main, so no upstream commit
#       past the nightly comes along) applied in manifest order, one commit
#       per branch that names the branch and its head. Stops at the first
#       conflict for a hand resolution; rerun with the same workdir after
#       committing it and it carries on. Then push the branch, and set
#       VLLM_REF, VLLM_BASE and BASE in the Dockerfile.
#
# docs/vllm-stack.md has the known conflicts and how they resolve.
set -euo pipefail
cd "$(dirname "$0")/.."

FORK=${FORK:-https://github.com/mmastrac/vllm.git}
UPSTREAM=${UPSTREAM:-https://github.com/vllm-project/vllm.git}
UPSTREAM_GH=vllm-project/vllm
MANIFEST=vllm-stack.tsv

rows() { grep -v '^#' "$MANIFEST" | grep -v '^$'; }
dockerfile_arg() { sed -n "s/^ARG $1=//p" Dockerfile | head -1; }

status() {
  local base; base=$(dockerfile_arg VLLM_BASE)
  echo "image base: ${base:0:10}"
  printf '%-30s %-12s %-10s %-9s %s\n' branch pinned fork PR verdict
  rows | while IFS=$'\t' read -r branch head pr parent; do
    local now fork state verdict in_base=""
    now=$(git ls-remote "$FORK" "refs/heads/$branch" | cut -f1)
    if [[ -z "$now" ]]; then fork=gone
    elif [[ "$now" == "$head" ]]; then fork=same
    else fork="${now:0:10}"; fi
    state=-
    if [[ "$pr" != - ]]; then
      state=$(gh pr view "$pr" -R "$UPSTREAM_GH" --json state -q .state 2>/dev/null || echo "?")
      if [[ "$state" == MERGED ]]; then
        local merge; merge=$(gh pr view "$pr" -R "$UPSTREAM_GH" --json mergeCommit -q .mergeCommit.oid)
        # "behind" or "identical": the merge commit is in the base's history.
        case $(gh api "repos/$UPSTREAM_GH/compare/$base...$merge" -q .status 2>/dev/null) in
          behind|identical) in_base=yes ;; *) in_base=no ;;
        esac
      fi
    fi
    if [[ "$in_base" == yes ]]; then verdict="drop: merged and in the base nightly"
    elif [[ "$state" == MERGED ]]; then verdict="merged upstream; drop once BASE moves past it"
    elif [[ "$state" == CLOSED ]]; then verdict="PR closed; decide whether to keep carrying it"
    elif [[ "$fork" == gone ]]; then verdict="branch deleted on the fork"
    elif [[ "$fork" != same ]]; then verdict="moved; rebuild the stack"
    else verdict=ok; fi
    printf '%-30s %-12s %-10s %-9s %s\n' "$branch" "${head:0:10}" "$fork" "$state" "$verdict"
  done
}

build() {
  local nightly=$1 dir=${2:-./.vllm-stack}
  if [[ ! -d "$dir/.git" ]]; then
    git clone --filter=blob:none --quiet "$FORK" "$dir"
    git -C "$dir" remote add upstream "$UPSTREAM"
  fi
  cd "$dir"
  git fetch --quiet upstream main "$nightly"
  git fetch --quiet origin
  if ! git rev-parse -q --verify djev-spark-stack-new >/dev/null; then
    git checkout --quiet -b djev-spark-stack-new "$nightly"
  else
    git checkout --quiet djev-spark-stack-new
  fi
  local log; log=$(git log --format=%B "$nightly..HEAD")
  rows_from() { grep -v '^#' "$OLDPWD/$MANIFEST" | grep -v '^$'; }
  rows_from | while IFS=$'\t' read -r branch head pr parent; do
    if grep -q "^stack: $branch ${head:0:10}" <<<"$log"; then
      echo "have $branch"; continue
    fi
    git merge-base --is-ancestor "$head" "origin/$branch" 2>/dev/null \
      || { echo "$branch: pinned head ${head:0:10} is not on the fork branch" >&2; exit 1; }
    # A fix stacked on another branch applies as its diff from that branch's
    # pinned head; any other branch as its diff from upstream main.
    local mb
    if [[ -n "${parent:-}" ]]; then
      mb=$(rows_from | awk -F'\t' -v p="$parent" '$1 == p {print $2}')
      [[ -n "$mb" ]] || { echo "$branch: parent $parent is not in the manifest" >&2; exit 1; }
    else
      mb=$(git merge-base "$head" upstream/main)
    fi
    echo "apply $branch ${head:0:10} (base ${mb:0:10})"
    if ! git diff "$mb" "$head" | git apply -3 --index; then
      echo "conflict in $branch: resolve, git add, then" >&2
      echo "  git commit -m 'stack: $branch ${head:0:10} (PR $pr)' && $0 build $nightly $dir" >&2
      exit 1
    fi
    git commit --quiet -m "stack: $branch ${head:0:10} (PR $pr)"
  done
  echo "stack built on ${nightly:0:10}: $(git rev-parse HEAD)"
}

case ${1:-status} in
  status) status ;;
  build) [[ $# -ge 2 ]] || { echo "usage: $0 build <nightly commit> [workdir]" >&2; exit 2; }; build "$2" "${3:-}" ;;
  *) echo "usage: $0 status | build <nightly commit> [workdir]" >&2; exit 2 ;;
esac
