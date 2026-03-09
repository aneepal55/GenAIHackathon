import DashboardClient from "../components/dashboard-client";
import { getDashboardData } from "../lib/data";

export const dynamic = "force-dynamic";

export default async function Page() {
  const data = await getDashboardData();
  return <DashboardClient data={data} />;
}
