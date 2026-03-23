from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class DataPanelItem:
    category: str
    name: str
    label: str


class DataPanelWidget(QtWidgets.QWidget):
    category_load_requested = QtCore.Signal(str)
    item_activated = QtCore.Signal(str, str)
    item_store_requested = QtCore.Signal(str, str)
    item_unload_requested = QtCore.Signal(str, str)
    header_clicked = QtCore.Signal()

    CATEGORY_ORDER = ("seismic", "attribute", "horizon", "scatter", "polygon", "model", "well")
    CATEGORY_LABELS = {
        "seismic": "Seismic",
        "attribute": "Attribute",
        "horizon": "Horizon",
        "scatter": "Scatter",
        "polygon": "Polygon",
        "model": "Model",
        "well": "Well",
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._category_items: dict[str, QtWidgets.QTreeWidgetItem] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.header_button = QtWidgets.QToolButton()
        self.header_button.setText("Data")
        self.header_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.header_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.header_button.clicked.connect(self.header_clicked.emit)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.header_button.setFont(font)
        layout.addWidget(self.header_button)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.itemClicked.connect(self._handle_item_clicked)
        self.tree.itemDoubleClicked.connect(self._handle_item_double_clicked)
        self.tree.customContextMenuRequested.connect(self._open_context_menu)
        layout.addWidget(self.tree, stretch=1)

        self._build_categories()

    def _build_categories(self) -> None:
        for category in self.CATEGORY_ORDER:
            item = QtWidgets.QTreeWidgetItem([self.CATEGORY_LABELS[category]])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "category", "category": category})
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            item.setExpanded(True)
            self.tree.addTopLevelItem(item)
            self._category_items[category] = item

    def set_items(self, items_by_category: dict[str, list[DataPanelItem]]) -> None:
        current = self.current_item()
        self.tree.blockSignals(True)
        for category in self.CATEGORY_ORDER:
            category_item = self._category_items[category]
            category_item.takeChildren()
            for entry in items_by_category.get(category, []):
                child = QtWidgets.QTreeWidgetItem([entry.label])
                child.setData(
                    0,
                    QtCore.Qt.ItemDataRole.UserRole,
                    {"kind": "data", "category": entry.category, "name": entry.name},
                )
                child.setToolTip(0, entry.name)
                category_item.addChild(child)
            category_item.setExpanded(True)
        self.tree.blockSignals(False)
        if current is not None:
            self.select_item(*current)

    def current_item(self) -> tuple[str, str] | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        payload = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict) or payload.get("kind") != "data":
            return None
        return str(payload["category"]), str(payload["name"])

    def select_item(self, category: str, name: str) -> None:
        root = self._category_items.get(category)
        if root is None:
            return
        for index in range(root.childCount()):
            child = root.child(index)
            payload = child.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(payload, dict) and payload.get("name") == name:
                self.tree.setCurrentItem(child)
                return

    def _handle_item_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        del column
        payload = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(payload, dict) and payload.get("kind") == "data":
            self.item_activated.emit(str(payload["category"]), str(payload["name"]))

    def _handle_item_double_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        self._handle_item_clicked(item, column)

    def _open_context_menu(self, position: QtCore.QPoint) -> None:
        item = self.tree.itemAt(position)
        if item is None:
            return
        payload = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return

        menu = QtWidgets.QMenu(self)
        if payload.get("kind") == "category":
            action = menu.addAction("Load")
            action.triggered.connect(lambda: self.category_load_requested.emit(str(payload["category"])))
        elif payload.get("kind") == "data":
            store_action = menu.addAction("Store")
            unload_action = menu.addAction("Unload")
            category = str(payload["category"])
            name = str(payload["name"])
            store_action.triggered.connect(lambda: self.item_store_requested.emit(category, name))
            unload_action.triggered.connect(lambda: self.item_unload_requested.emit(category, name))
        if menu.isEmpty():
            return
        menu.exec(self.tree.viewport().mapToGlobal(position))


DataPanelWindow = DataPanelWidget


class ProjectPanelWidget(QtWidgets.QWidget):
    header_clicked = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.header_button = QtWidgets.QToolButton()
        self.header_button.setText("Project")
        self.header_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.header_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.header_button.clicked.connect(self.header_clicked.emit)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.header_button.setFont(font)
        layout.addWidget(self.header_button)

        self.info_tree = QtWidgets.QTreeWidget()
        self.info_tree.setHeaderLabels(["Field", "Value"])
        self.info_tree.setRootIsDecorated(False)
        self.info_tree.setAlternatingRowColors(False)
        self.info_tree.header().setStretchLastSection(True)
        self.info_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.info_tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.info_tree, stretch=1)

    def set_info(self, entries: list[tuple[str, str]]) -> None:
        self.info_tree.clear()
        for label, value in entries:
            item = QtWidgets.QTreeWidgetItem([label, value])
            item.setToolTip(0, label)
            item.setToolTip(1, value)
            self.info_tree.addTopLevelItem(item)
