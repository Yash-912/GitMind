from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .github_collector import IssueRecord, PullRequestRecord
from .git_collector import CommitRecord


@dataclass(frozen=True)
class LinkRecord:
    link_id: str
    source_type: str
    source_id: str
    target_type: str
    target_id: str
    relation: str


def _link_id(
    source_type: str, source_id: str, target_type: str, target_id: str, relation: str
) -> str:
    return f"{source_type}:{source_id}->{target_type}:{target_id}:{relation}"


class CrossReferenceLinker:
    issue_pattern = re.compile(r"#(\d+)")

    def link_prs_to_issues(
        self, prs: Iterable[PullRequestRecord], issues: Iterable[IssueRecord]
    ) -> list[LinkRecord]:
        issue_map = {str(issue.number): issue for issue in issues}
        links: list[LinkRecord] = []
        for pr in prs:
            matches = self.issue_pattern.findall(pr.body or "") + self.issue_pattern.findall(
                pr.title or ""
            )
            for issue_id in set(matches):
                if issue_id in issue_map:
                    links.append(
                        LinkRecord(
                            link_id=_link_id(
                                "pr", str(pr.number), "issue", issue_id, "mentions"
                            ),
                            source_type="pr",
                            source_id=str(pr.number),
                            target_type="issue",
                            target_id=issue_id,
                            relation="mentions",
                        )
                    )
        return links

    def link_commits_to_prs(
        self, commits: Iterable[CommitRecord], prs: Iterable[PullRequestRecord]
    ) -> list[LinkRecord]:
        pr_by_merge = {
            pr.merge_commit_sha: pr for pr in prs if pr.merge_commit_sha is not None
        }
        links: list[LinkRecord] = []
        for commit in commits:
            if commit.sha in pr_by_merge:
                pr = pr_by_merge[commit.sha]
                links.append(
                    LinkRecord(
                        link_id=_link_id(
                            "commit",
                            commit.sha,
                            "pr",
                            str(pr.number),
                            "merge_commit",
                        ),
                        source_type="commit",
                        source_id=commit.sha,
                        target_type="pr",
                        target_id=str(pr.number),
                        relation="merge_commit",
                    )
                )
        return links
