from pathlib import Path
import unittest

from tools.validate_documentation import validate_links

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DocumentationContractTests(unittest.TestCase):
    def test_readme_leads_with_portfolio_and_uses_source_development(self):
        readme = (PROJECT_ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertLessEqual(len(readme.splitlines()), 220)
        for phrase in ('five-site healthcare', '150 messages per day', 'docs/images/portal-demo.png',
                       'Deterministic verification', "python -m pip install -e '.[dev]'"):
            self.assertIn(phrase, readme)
        self.assertNotIn('gh release download', readme)
        self.assertLess(readme.index('Deployment context'), readme.index('Development and tests'))

    def test_all_portfolio_links_resolve(self):
        self.assertEqual(validate_links(PROJECT_ROOT), [])

    def test_missing_targets_and_anchors_are_reported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('# Overview\n[Missing](missing.md)\n[Anchor](#absent)\n')
            self.assertEqual(len(validate_links(root)), 2)
